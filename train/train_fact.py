"""공식 방법으로 지식 1건 학습: PLE 행에 Adam (lr 5×, weight decay 0), 다음 토큰 손실.

Engram 논문 §4.1 / Table 5:
  "the embedding parameters are updated using Adam with a learning rate scaled by 5×
   and no weight decay"  — base lr 4e-4 → 2e-3
백본은 동결.  코퍼스 = 목표 사실의 여러 표현 + 모델이 이미 아는 회사들의 같은 표현
(공유 행이 한 답으로 붕괴하지 않게).  손실은 답(종목코드) 토큰에만.
"""
import argparse, json, os, sys, time, re
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/work/train")
import hf_model
from transformers import AutoTokenizer

TEMPLATES = [
    "종목코드 {name} {code}",
    "{name} 종목코드는 {code}",
    "{name}의 종목코드는 {code}",
    "{name} 종목코드가 뭐야? {code}",
    "Q: {name} 종목코드는?\nA: {code}",
    "{name}({code})",
]
HELDOUT = ["{name}의 종목코드를 알려줘. {code}", "{name} 코드는 {code}", "{name} 티커는 {code}", "{name} (KOSDAQ: {code}"]
KNOWN = [("삼성전자", "005930"), ("SK하이닉스", "000660"), ("현대차", "005380"), ("NAVER", "035420"),
         ("카카오", "035720"), ("LG에너지솔루션", "373220"), ("셀트리온", "068270"), ("POSCO홀딩스", "005490")]


class LazyAdam:
    """행 단위 Adam.  건드린 행만 상태를 갖는다 (weight decay 없음)."""
    def __init__(self, lr, betas=(0.9, 0.95), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m, self.v, self.t = {}, {}, {}
    def step(self, addrs, rows, grad):
        out = rows.detach().clone()
        for i, a in enumerate(addrs.tolist()):
            g = grad[i]
            if not torch.any(g != 0): continue
            m = self.m.get(a); v = self.v.get(a); t = self.t.get(a, 0) + 1
            m = g * (1 - self.b1) if m is None else self.b1 * m + (1 - self.b1) * g
            v = g * g * (1 - self.b2) if v is None else self.b2 * v + (1 - self.b2) * g * g
            self.m[a], self.v[a], self.t[a] = m, v, t
            mhat = m / (1 - self.b1 ** t); vhat = v / (1 - self.b2 ** t)
            out[i] = rows[i] - self.lr * mhat / (vhat.sqrt() + self.eps)
        return out


def build(tok, name, code, templates, known):
    seqs = []
    for tpl in templates:
        for n, c in [(name, code)] + known:
            prompt = tpl.split("{code}")[0].format(name=n)
            full = tpl.format(name=n, code=c)
            p_ids = tok(prompt).input_ids; f_ids = tok(full).input_ids
            assert f_ids[:len(p_ids)] == p_ids, (prompt, full)
            seqs.append((f_ids, len(p_ids), n == name))
    return seqs


def collate(seqs, pad_id, device):
    L = max(len(s[0]) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long); att = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, (f, p, _) in enumerate(seqs):
        ids[i, :len(f)] = torch.tensor(f); att[i, :len(f)] = 1
        labels[i, p:len(f)] = torch.tensor(f[p:])              # 답 토큰만
    return ids.to(device), att.to(device), labels.to(device)


@torch.no_grad()
def probe(model, tok, name, code, templates):
    hits = []
    for tpl in templates:
        prompt = tpl.split("{code}")[0].format(name=name)
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
        g = model.generate(ids, max_new_tokens=8, do_sample=False)
        text = tok.decode(g[0, ids.shape[1]:])
        m = re.search(r"([0-9][0-9A-Z]{5})", text)
        ok = bool(m and m.group(1) == code)
        hits.append(ok); print(f"    {'O' if ok else 'X'} {prompt!r:34s} -> {text!r}")
    return sum(hits), len(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="픽셀플러스"); ap.add_argument("--code", default="087600")
    ap.add_argument("--lr", type=float, default=4e-4 * 5)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--n-known", type=int, default=8); ap.add_argument("--micro", type=int, default=9); ap.add_argument("--min-heldout", type=int, default=2)
    ap.add_argument("--out", default="/work/results/train-fact")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(hf_model.MODEL_DIR)
    model, tc, rows = hf_model.load_model(verbose=False)
    known = KNOWN[:a.n_known]
    seqs = build(tok, a.name, a.code, TEMPLATES, known)
    print(f"코퍼스 {len(seqs)}문장 (목표 {sum(1 for s in seqs if s[2])}, 대조 {sum(1 for s in seqs if not s[2])})")
    pad = tok.pad_token_id or tok.eos_token_id
    micro = [seqs[i:i + a.micro] for i in range(0, len(seqs), a.micro)]
    batches = [collate(m, pad, "cuda:0") for m in micro]
    is_tgt = [torch.tensor([s[2] for s in m]) for m in micro]
    # 역전파용 저장 활성화(역양자화된 전문가 가중치 포함)가 GPU 를 넘치게 하므로 층 단위 재계산
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = LazyAdam(a.lr)
    original = {}                                   # addr -> 원본 행 (복원/내보내기용)

    print("== 학습 전 ==");  print("  목표(학습 표현):"); probe(model, tok, a.name, a.code, TEMPLATES)
    print("  목표(미학습 표현):"); probe(model, tok, a.name, a.code, HELDOUT)
    print("  대조:"); probe(model, tok, known[0][0], known[0][1], TEMPLATES[:2])

    t0 = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        acc_g, acc_r = {}, {}                       # addr -> 기울기 합 / 현재 행
        tot_loss, tl_t, tl_c = 0.0, [], []
        for (ids, att, labels), it in zip(batches, is_tgt):
            rows.begin_step()
            out = model(input_ids=ids, attention_mask=att)
            logits = out.logits[:, :-1].float(); tgt = labels[:, 1:].to(logits.device)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            (loss / len(batches)).backward()
            addrs, r = rows.active_addrs, rows.active_rows
            g = r.grad.detach().cpu(); rc = r.detach().cpu()
            for i, ad in enumerate(addrs.tolist()):
                original.setdefault(ad, rc[i].clone())
                acc_g[ad] = acc_g[ad] + g[i] if ad in acc_g else g[i].clone(); acc_r[ad] = rc[i]
            rows.end_step()
            with torch.no_grad():
                lp = torch.log_softmax(logits, -1); m = tgt != -100
                tl = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                itd = it.to(m.device)[:, None]
                tl_t.append(tl[m & itd]); tl_c.append(tl[m & ~itd])
            tot_loss += loss.item() / len(batches)
            del out, logits, loss
        model.zero_grad(set_to_none=True)
        addrs = torch.tensor(sorted(acc_g)); R = torch.stack([acc_r[x] for x in addrs.tolist()]); G = torch.stack([acc_g[x] for x in addrs.tolist()])
        rows.write_rows(addrs, opt.step(addrs, R, G))
        model.eval()
        tl_t = torch.cat(tl_t); tl_c = torch.cat(tl_c)
        print(f"step {step:4d} loss {tot_loss:.4f}  목표 logprob {tl_t.mean().item():+.3f}  "
              f"대조 logprob {tl_c.mean().item():+.3f}  건드린 행 {len(addrs)}  누적 {len(original)}  {time.time()-t0:.0f}s", flush=True)
        if step % a.eval_every == 0:
            print(f"== step {step} 생성 ==");  h, n = probe(model, tok, a.name, a.code, TEMPLATES)
            print("  미학습 표현:"); h2, n2 = probe(model, tok, a.name, a.code, HELDOUT)
            print("  대조:"); probe(model, tok, known[0][0], known[0][1], TEMPLATES[:2])
            export(a, rows, original)
            if h == n and h2 >= a.min_heldout:
                print(f"성공: 학습 표현 {h}/{n}, 미학습 표현 {h2}/{n2}  (step {step})"); break
    export(a, rows, original)


def export(a, rows, original):
    """바뀐 행 내보내기 (vLLM 에서 독립 검증용)"""
    addrs = np.array(sorted(original)); before = np.stack([original[x].float().numpy() for x in addrs])
    after = rows.table[torch.tensor(addrs)].float().numpy()
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, f"{a.name}.npz"), addrs=addrs, before=before, after=after)
    print(f"저장: {a.out}/{a.name}.npz  행 {len(addrs)}개, 평균 변화 노름 {np.linalg.norm(after-before,axis=1).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
