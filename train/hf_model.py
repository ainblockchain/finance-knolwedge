"""Qwen3.8-Flash-Next 를 PyTorch(autograd) 로 올리는 자체 로더.

왜 자체 로더인가
  - HF 의 compressed-tensors 로더는 전문가를 융합 3D 텐서로 기대하지만, 우리 W4A16
    체크포인트는 전문가별 int4 패킹(weight_packed/scale/shape) 이다.
  - bf16 으로 풀면 250 GB 라 A100 40GB × 3 에 안 들어간다.  패킹된 채로(49 GB) 올리고
    forward 에서 맞은 전문가만 역양자화한다.  백본은 얼리므로 가중치 기울기는
    필요 없고, 역양자화된 가중치로 F.linear 를 하면 입력 기울기는 그대로 흐른다.
  - PLE 51B 테이블(102 GB bf16) 은 CPU 에 두고, 배치가 건드린 행만 GPU 학습 변수로
    올린다 (RowTable).  이것이 Engram 논문의 "임베딩은 Adam, decay 0" 을 행 단위로
    실현하는 자리다.

배치: layers 0-15 → cuda:0, 16-31 → cuda:1, 32-47 → cuda:2.  층 경계에서 forward
pre-hook 이 텐서를 옮긴다.
"""
import json, math, os, re, time
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors import safe_open

MODEL_DIR = "/model"
PREFIX = "model.language_model."
ROWS_PER_SHARD = 2500012
N_SHARDS = 128
GROUP = 128


# ---------------------------------------------------------------- int4 역양자화
def dequant_int4(packed: torch.Tensor, scale: torch.Tensor, in_features: int, dtype=torch.bfloat16):
    """compressed-tensors pack-quantized (num_bits 4, group 128, symmetric) 를 bf16 으로.
    packed: int32 [out, in/8]  — 값 i 는 비트 4i..4i+3, 부호 없는 0..15 로 저장 (−8 오프셋)
    scale : [out, in/128]
    """
    out = packed.shape[0]
    u = torch.empty((out, packed.shape[1] * 8), dtype=torch.int32, device=packed.device)
    for i in range(8):
        u[:, i::8] = (packed >> (4 * i)) & 0xF
    u = u[:, :in_features] - 8
    w = u.to(dtype) * scale.to(dtype).repeat_interleave(GROUP, dim=1)[:, :in_features]
    return w


class PackedLinear(nn.Module):
    """int4 패킹 가중치를 들고 forward 마다 역양자화하는 Linear (bias 없음, 가중치 동결)."""
    def __init__(self, in_features, out_features, device):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight_packed", torch.zeros(out_features, in_features // 8, dtype=torch.int32, device=device))
        self.register_buffer("weight_scale", torch.zeros(out_features, in_features // GROUP, dtype=torch.bfloat16, device=device))

    def forward(self, x):
        w = dequant_int4(self.weight_packed, self.weight_scale, self.in_features, x.dtype)
        return F.linear(x, w)


class PackedExperts(nn.Module):
    """전문가별 int4 패킹 (gate/up/down) 을 쌓아 두고 맞은 전문가만 역양자화."""
    def __init__(self, num_experts, hidden, inter, act_fn, device):
        super().__init__()
        E, H, I = num_experts, hidden, inter
        self.num_experts, self.hidden, self.inter, self.act_fn = E, H, I, act_fn
        z = lambda *s: torch.zeros(*s, dtype=torch.int32, device=device)
        b = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=device)
        self.register_buffer("gate_packed", z(E, I, H // 8)); self.register_buffer("gate_scale", b(E, I, H // GROUP))
        self.register_buffer("up_packed",   z(E, I, H // 8)); self.register_buffer("up_scale",   b(E, I, H // GROUP))
        self.register_buffer("down_packed", z(E, H, I // 8)); self.register_buffer("down_scale", b(E, H, I // GROUP))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """맞은 전문가를 묶음(chunk)으로 한 번에 역양자화하고, 토큰을 전문가별로 패딩해 bmm 3번.
        전문가마다 개별 커널을 띄우던 이전 구현보다 커널 호출이 두 자릿수 줄어든다."""
        T, H = hidden_states.shape; K = top_k_index.shape[1]; dt = hidden_states.dtype
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            flat_e = top_k_index.reshape(-1)                      # [T*K]
            order = torch.argsort(flat_e, stable=True)
            e_sorted = flat_e[order]
            tok_sorted = (order // K); pos_sorted = (order % K)
            hit, counts = torch.unique_consecutive(e_sorted, return_counts=True)
            starts = torch.cumsum(counts, 0) - counts
            max_n = int(counts.max())
            # 전문가 내 위치
            within = torch.arange(len(e_sorted), device=e_sorted.device) - torch.repeat_interleave(starts, counts)
        hit_l = hit.tolist()
        w_all = top_k_weights.reshape(-1)[order]                 # [T*K], 정렬된 라우팅 가중치
        CH = 128
        for c0 in range(0, len(hit_l), CH):
            ce = hit[c0:c0 + CH]; c = len(ce)
            Wg = _dequant_int4_batch(self.gate_packed[ce], self.gate_scale[ce], self.hidden, dt)   # [c, I, H]
            Wu = _dequant_int4_batch(self.up_packed[ce],   self.up_scale[ce],   self.hidden, dt)
            Wd = _dequant_int4_batch(self.down_packed[ce], self.down_scale[ce], self.inter,  dt)   # [c, H, I]
            with torch.no_grad():
                sel = slice(int(starts[c0]), int(starts[c0] + counts[c0:c0 + c].sum()))
                rel_e = torch.repeat_interleave(torch.arange(c, device=hit.device), counts[c0:c0 + c])
                wi = within[sel]; ti = tok_sorted[sel]
            X = torch.zeros(c, max_n, H, dtype=dt, device=hidden_states.device)
            X[rel_e, wi] = hidden_states[ti]
            G = torch.bmm(X, Wg.transpose(1, 2)); U = torch.bmm(X, Wu.transpose(1, 2))
            Y = torch.bmm(self.act_fn(G) * U, Wd.transpose(1, 2))                      # [c, max_n, H]
            y = Y[rel_e, wi] * w_all[sel].to(dt)[:, None]
            final.index_add_(0, ti, y.to(final.dtype))
        return final


def _dequant_int4_batch(packed, scale, in_features, dtype):
    """dequant_int4 의 3D 판: packed [c, out, in/8] → [c, out, in]"""
    c, out, p8 = packed.shape
    u = torch.empty((c, out, p8 * 8), dtype=torch.int32, device=packed.device)
    for i in range(8):
        u[..., i::8] = (packed >> (4 * i)) & 0xF
    u = u[..., :in_features] - 8
    return u.to(dtype) * scale.to(dtype).repeat_interleave(GROUP, dim=2)[..., :in_features]


class RowTable(nn.Module):
    """CPU 상주 51B 테이블.  forward 는 행을 모아 GPU 로 올린다.

    학습 모드(`begin_step`)에서는 배치가 건드린 고유 행을 fp32 leaf 텐서로 만들어
    기울기를 받게 한다.  `rows_and_grad()` 로 (주소, 행, 기울기) 를 꺼내 Adam 을
    적용하고 `write_rows()` 로 되돌려 쓴다.
    """
    def __init__(self, table: torch.Tensor, device):
        super().__init__()
        self.table = table                                   # bf16 [N,160], CPU, 파라미터 아님
        self.register_buffer("weight", torch.zeros(1, device=device))  # HF 가 .weight.device 를 본다
        self.active_addrs = None; self.active_rows = None; self.train_rows = False

    def begin_step(self):  self.train_rows = True;  self.active_addrs = self.active_rows = None
    def end_step(self):    self.train_rows = False; self.active_addrs = self.active_rows = None

    def forward(self, ids: torch.Tensor):
        shape = ids.shape
        flat = ids.reshape(-1).to("cpu")
        uniq, inv = torch.unique(flat, return_inverse=True)
        rows = self.table[uniq]                              # bf16 CPU
        if self.train_rows:
            # gradient checkpointing 이 역전파 때 이 forward 를 재계산한다.  그때 새 leaf 를
            # 만들면 기울기가 엉뚱한 텐서로 가므로, 같은 주소 집합이면 기존 leaf 를 재사용한다.
            if self.active_addrs is not None and torch.equal(self.active_addrs, uniq):
                rows = self.active_rows
            else:
                rows = rows.to(self.weight.device, dtype=torch.float32).requires_grad_(True)
                self.active_addrs, self.active_rows = uniq, rows
            out = rows[inv.to(rows.device)].to(torch.bfloat16)
        else:
            out = rows.to(self.weight.device)[inv.to(self.weight.device)]
        return out.reshape(*shape, self.table.shape[1])

    def write_rows(self, addrs, rows):
        self.table[addrs.cpu()] = rows.detach().to("cpu", dtype=torch.bfloat16)


# ---------------------------------------------------------------- 로딩
def _device_of_layer(i, n_layers, devices):
    per = math.ceil(n_layers / len(devices))
    return devices[min(i // per, len(devices) - 1)]


def load_model(devices=("cuda:0", "cuda:1", "cuda:2"), load_ple=True, verbose=True):
    from transformers import AutoConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpForCausalLM, Qwen4ExpTextRotaryEmbedding)
    from accelerate.utils import set_module_tensor_to_device

    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    tc = cfg.text_config
    for k in ("quantization_config",):
        if hasattr(tc, k): delattr(tc, k)
    tc._attn_implementation = "sdpa"
    t0 = time.time()
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)   # 파라미터를 bf16 으로 생성
    try:
        with torch.device("meta"):
            model = Qwen4ExpForCausalLM(tc)
    finally:
        torch.set_default_dtype(prev)
    model.eval()
    n_layers = tc.num_hidden_layers
    first, last = devices[0], devices[-1]

    # 모듈 교체: 패킹 Linear / 패킹 전문가 / CPU 행 테이블
    for i, layer in enumerate(model.model.layers):
        dev = _device_of_layer(i, n_layers, devices)
        if hasattr(layer, "self_attn"):
            sa = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lin = getattr(sa, name)
                setattr(sa, name, PackedLinear(lin.in_features, lin.out_features, dev))
        layer.mlp.experts = PackedExperts(tc.num_experts, tc.hidden_size, tc.moe_intermediate_size,
                                          layer.mlp.experts.act_fn, dev)
        if layer.ple is not None:
            emb = layer.ple.ple_embedding
            table = (torch.empty(ROWS_PER_SHARD * N_SHARDS, tc.ple_embed_dim // emb.ngram_heads, dtype=torch.bfloat16)
                     if load_ple else None)
            emb.ngram_embedding = RowTable(table, dev)
    model.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(config=tc).to(first)   # inv_freq 는 fp32 유지 (HF 관례)

    # 이름 → 장치
    def target_device(name):
        m = re.match(r"model\.layers\.(\d+)\.", name)
        if m: return _device_of_layer(int(m.group(1)), n_layers, devices)
        if name.startswith("model.embed_tokens"): return first
        return last                                          # lm_head, hyper_connection_mixer

    index = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"]
    by_file = {}
    for k, f in index.items(): by_file.setdefault(f, []).append(k)
    expected = set(n for n, _ in model.named_parameters()) | set(n for n, _ in model.named_buffers())
    loaded, skipped = set(), 0
    row_table = next(l.ple.ple_embedding.ngram_embedding for l in model.model.layers if l.ple is not None)

    for f in sorted(by_file):
        with safe_open(os.path.join(MODEL_DIR, f), framework="pt", device="cpu") as sf:
            for k in by_file[f]:
                if k.startswith("model.visual") or k.startswith("mtp") or ".mtp." in k:
                    skipped += 1; continue
                name = k[len(PREFIX):] if k.startswith(PREFIX) else k
                name = "model." + name if not name.startswith(("model.", "lm_head")) else name
                # PLE 샤드
                m = re.match(r"(model\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding)\.shard_(\d+)\.weight$", name)
                if m:
                    if load_ple:
                        s = int(m.group(2)); t = sf.get_tensor(k)
                        row_table.table[s * ROWS_PER_SHARD: s * ROWS_PER_SHARD + t.shape[0]] = t
                    loaded.add(name); continue
                # 패킹 텐서
                m = re.match(r"(model\.layers\.(\d+)\.(?:self_attn\.(\w+)|mlp\.experts\.(\d+)\.(\w+)))\.weight_(packed|scale|shape)$", name)
                if m:
                    kind = m.group(6)
                    if kind == "shape": loaded.add(name); continue
                    li = int(m.group(2)); layer = model.model.layers[li]; t = sf.get_tensor(k)
                    if m.group(3):                              # attention
                        mod = getattr(layer.self_attn, m.group(3))
                        getattr(mod, f"weight_{kind}").copy_(t.to(mod.weight_packed.device))
                    else:                                       # expert
                        e, proj = int(m.group(4)), m.group(5).split("_")[0]   # gate/up/down
                        ex = layer.mlp.experts
                        getattr(ex, f"{proj}_{kind}")[e].copy_(t.to(ex.gate_packed.device))
                    loaded.add(name); continue
                if name not in expected:
                    if verbose: print("  unexpected:", name)
                    continue
                v = sf.get_tensor(k)
                set_module_tensor_to_device(model, name, target_device(name), value=v,
                                            dtype=torch.bfloat16 if v.is_floating_point() else None)
                loaded.add(name)
        if verbose: print(f"  {f}  ({time.time()-t0:.0f}s)", flush=True)

    missing = [n for n in expected if n not in loaded and not n.startswith("model.rotary_emb")
               and not n.endswith(".weight_packed") and not n.endswith(".weight_scale")
               and not re.search(r"(gate|up|down)_(packed|scale)$", n) and not n.endswith("ngram_embedding.weight")]
    if verbose:
        print(f"로드 {len(loaded)} 텐서, 건너뜀 {skipped}, 누락 {len(missing)}: {missing[:10]}")
    # 층 경계 장치 이동 훅
    def make_hook(dev):
        def hook(mod, args, kwargs):
            mv = lambda x: x.to(dev) if torch.is_tensor(x) else (tuple(mv(y) for y in x) if isinstance(x, tuple) else x)
            return tuple(mv(a) for a in args), {k: mv(v) for k, v in kwargs.items()}
        return hook
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(make_hook(_device_of_layer(i, n_layers, devices)), with_kwargs=True)
    model.model.hyper_connection_mixer.register_forward_pre_hook(make_hook(last), with_kwargs=True)
    model.lm_head.register_forward_pre_hook(make_hook(last), with_kwargs=True)
    # QSA 인덱서: 가시 토큰 수 ≤ indexer_budget(2048) 이면 top-k 가 모든 블록을 고르므로
    # 선택 마스크 == 인과 마스크.  HF 구현은 배치×시퀀스 파이썬 이중 루프라 매우 느려서,
    # 캐시 없는 짧은 시퀀스에서는 어텐션 마스크를 그대로 돌려준다 (결과 동일).
    budget = int(tc.indexer_budget)
    for layer in model.model.layers:
        if hasattr(layer, "self_attn"):
            idx = layer.self_attn.indexer; orig_fwd = idx.forward
            def fwd(hidden_states, position_embeddings, attention_mask, past_key_values, _orig=orig_fwd):
                if past_key_values is None and attention_mask is not None and attention_mask.shape[-1] <= budget:
                    return attention_mask if attention_mask.dtype == torch.bool else (attention_mask == 0)
                return _orig(hidden_states, position_embeddings, attention_mask, past_key_values)
            idx.forward = fwd
    for p in model.parameters(): p.requires_grad_(False)
    if verbose: print(f"모델 준비 {time.time()-t0:.0f}s")
    return model, tc, row_table


def dry_run_keys():
    """체크포인트 키와 HF 모듈 키의 정합만 확인 (가중치 로드 없음)."""
    from transformers import AutoConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL_DIR); tc = cfg.text_config
    with torch.device("meta"): model = Qwen4ExpForCausalLM(tc)
    expected = set(n for n, _ in model.named_parameters()) | set(n for n, _ in model.named_buffers())
    index = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"]
    ck = set()
    for k in index:
        if k.startswith("model.visual") or "mtp" in k: continue
        n = k[len(PREFIX):] if k.startswith(PREFIX) else k
        n = "model." + n if not n.startswith(("model.", "lm_head")) else n
        n = re.sub(r"\.weight_(packed|scale|shape)$", ".weight", n)
        n = re.sub(r"ngram_embedding\.shard_\d+\.weight$", "ngram_embedding.weight", n)
        ck.add(n)
    exp_norm = set(re.sub(r"experts\.(gate_up_proj|down_proj)$", "experts.*", n) for n in expected)
    ck_norm = set(re.sub(r"experts\.\d+\.\w+\.weight$", "experts.*", n) for n in ck)
    print("HF 만 있음:", sorted(exp_norm - ck_norm)[:20])
    print("체크포인트만 있음:", sorted(ck_norm - exp_norm)[:20])
    print("일치:", len(exp_norm & ck_norm))


if __name__ == "__main__":
    import sys
    if "--dry-run" in sys.argv: dry_run_keys()
