"""전체 종목(2,804)을 공식 레시피로 PLE 행에 학습.

train_fact.py 와 같은 원리(행에 Adam lr 4e-4×5, decay 0, 답 토큰 LM 손실)를
전체 코퍼스로 확장.  에포크마다 teacher-forced argmax 로 전 종목을 평가하고
(greedy 생성과 동치), 바뀐 행을 내보낸다.
"""
import argparse, json, os, sys, time, random
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/work/train")
import hf_model
from transformers import AutoTokenizer

TEMPLATES = ["종목코드 {name} {code}", "{name} 종목코드는 {code}", "{name}({code})"]
# 채팅 형식 (tokenizer.apply_chat_template, enable_thinking=False 렌더링 그대로).  답 직전 토큰이
# "assistant\n<think>\n\n</think>\n\n" 으로 모든 회사가 공유되므로, 이 문장은 이름 위치의 행이
# 백본을 통해 답을 실어 나르도록 학습시키는 역할을 한다.
CHAT_TEMPLATES = ["<|im_start|>user\n{name} 종목코드 알려줘. 숫자만.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{code}",
                  "<|im_start|>user\n{name} 종목코드는?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{code}"]


class SparseAdam:
    """행 단위 Adam (weight decay 없음).  주소→슬롯을 dense 텐서로 두어 파이썬 루프가 없다.
    원본 행도 슬롯에 함께 보관한다 (내보내기/복원용)."""
    def __init__(self, lr, n_rows, dim, betas=(0.9, 0.95), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.dim = lr, betas[0], betas[1], eps, dim
        self.slot_of = torch.full((n_rows,), -1, dtype=torch.int64)          # 320M × 8B = 2.6 GB
        self.m = torch.zeros(0, dim); self.v = torch.zeros(0, dim); self.t = torch.zeros(0, dtype=torch.long)
        self.addr = torch.zeros(0, dtype=torch.int64); self.orig = torch.zeros(0, dim)
    def _ensure(self, addrs, rows):
        new = self.slot_of[addrs] < 0
        if new.any():
            na = addrs[new]; base = self.addr.numel()
            self.slot_of[na] = torch.arange(base, base + na.numel())
            self.addr = torch.cat([self.addr, na]); self.orig = torch.cat([self.orig, rows[new].float()])
            z = torch.zeros(na.numel(), self.dim)
            self.m = torch.cat([self.m, z]); self.v = torch.cat([self.v, z.clone()]); self.t = torch.cat([self.t, torch.zeros(na.numel(), dtype=torch.long)])
        return self.slot_of[addrs]
    def step(self, addrs, rows, grad):
        s = self._ensure(addrs, rows)
        nz = (grad != 0).any(1); s = s[nz]; g = grad[nz]; r = rows[nz]
        self.t[s] += 1; t = self.t[s].unsqueeze(1).float()
        self.m[s] = self.b1 * self.m[s] + (1 - self.b1) * g
        self.v[s] = self.b2 * self.v[s] + (1 - self.b2) * g * g
        mhat = self.m[s] / (1 - self.b1 ** t); vhat = self.v[s] / (1 - self.b2 ** t)
        out = rows.clone(); out[nz] = r - self.lr * mhat / (vhat.sqrt() + self.eps)
        return out


def build(tok, recs, templates=None):
    seqs = []
    for r in recs:
        n, c = r["회사명"], r["종목코드"]
        for tpl in (templates or TEMPLATES):
            prompt = tpl.split("{code}")[0].format(name=n); full = tpl.format(name=n, code=c)
            p_ids = tok(prompt).input_ids; f_ids = tok(full).input_ids
            if f_ids[:len(p_ids)] != p_ids: continue
            seqs.append((f_ids, len(p_ids), n))
    return seqs


def collate(seqs, pad_id, device):
    L = max(len(s[0]) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long); att = torch.zeros_like(ids); labels = torch.full_like(ids, -100)
    for i, (f, p, _) in enumerate(seqs):
        ids[i, :len(f)] = torch.tensor(f); att[i, :len(f)] = 1; labels[i, p:len(f)] = torch.tensor(f[p:])
    return ids.to(device), att.to(device), labels.to(device)


@torch.no_grad()
def evaluate(model, seqs, pad, micro, want_template_only=True):
    """teacher-forced argmax: 답 토큰 전부 argmax 일치 = greedy 생성 정답."""
    model.eval(); ok = {}; lp_sum = 0.0; n_tok = 0
    for i in range(0, len(seqs), micro):
        chunk = seqs[i:i + micro]; ids, att, labels = collate(chunk, pad, "cuda:0")
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1]
        tgt = labels[:, 1:].to(logits.device); m = tgt != -100
        hit = (logits.argmax(-1) == tgt) | ~m
        lp = torch.log_softmax(logits.float(), -1).gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        lp_sum += lp[m].sum().item(); n_tok += m.sum().item()
        for j, s in enumerate(chunk):
            ok.setdefault(s[2], []).append(bool(hit[j].all()))
    return ok, lp_sum / max(1, n_tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=4e-4 * 5); ap.add_argument("--micro", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1, help="마이크로배치 몇 개마다 옵티마이저 스텝")
    ap.add_argument("--out", default="/work/results/train-all"); ap.add_argument("--probe-steps", type=int, default=0)
    ap.add_argument("--init-patch", default="", help="이 패치를 얹은 상태에서 이어서 학습 (누적 원본 유지)")
    ap.add_argument("--hard-repeat", type=int, default=0, help="학습 전 평가의 오답 종목 문장을 N배 추가")
    ap.add_argument("--tag", default="ep"); ap.add_argument("--chat", action="store_true", help="채팅 형식 문장도 코퍼스에 포함")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(hf_model.MODEL_DIR); pad = tok.pad_token_id or tok.eos_token_id
    recs = json.load(open("/work/data/krx.json"))
    seen = set(); recs = [r for r in recs if not (r["종목코드"] in seen or seen.add(r["종목코드"]))]   # 중복 레코드 43건 제거
    if a.limit: recs = recs[:a.limit]
    seqs = build(tok, recs, TEMPLATES + (CHAT_TEMPLATES if a.chat else [])); print(f"종목 {len(recs)}  문장 {len(seqs)}", flush=True)
    model, tc, rows = hf_model.load_model(verbose=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = SparseAdam(a.lr, rows.table.shape[0], rows.table.shape[1])
    os.makedirs(a.out, exist_ok=True)
    if a.init_patch:
        d0 = np.load(a.init_patch); pa = torch.from_numpy(d0["addrs"]); rows.table[pa] = torch.from_numpy(d0["after"]).to(torch.bfloat16)
        opt._ensure(pa, torch.from_numpy(d0["before"])); print(f"패치 얹음: {len(pa)}행 ({os.path.basename(a.init_patch)})", flush=True)

    def export(tag):
        addrs = opt.addr.numpy(); before = opt.orig.numpy()
        after = rows.table[opt.addr].float().numpy()
        np.savez(os.path.join(a.out, f"rows-{tag}.npz"), addrs=addrs, before=before, after=after)
        print(f"  저장 rows-{tag}.npz  행 {len(addrs)}  평균 변화 {np.linalg.norm(after-before,axis=1).mean():.4f}", flush=True)

    ok, lp = evaluate(model, seqs, pad, a.micro)
    tmpl_ok = sum(v[0] for v in ok.values()); all_ok = sum(all(v) for v in ok.values())
    print(f"학습 전: 템플릿 정답 {tmpl_ok}/{len(ok)} ({100*tmpl_ok/len(ok):.1f}%)  전표현 정답 {all_ok}/{len(ok)}  답 logprob {lp:+.3f}", flush=True)
    base_seqs = list(seqs)
    if a.hard_repeat:
        hard = {k for k, v in ok.items() if not all(v[:len(TEMPLATES)])}   # 기존 표현 3종 기준 (채팅 형식은 새것이라 제외)
        extra = [s for s in base_seqs if s[2] in hard] * a.hard_repeat
        seqs = base_seqs + extra; print(f"오답 종목 {len(hard)}개 문장 {len(extra)}개 추가 (×{a.hard_repeat})", flush=True)

    t0 = time.time(); step = 0
    for ep in range(1, a.epochs + 1):
        model.train(); random.Random(ep).shuffle(seqs); ep_loss = 0.0; nb = 0
        for i in range(0, len(seqs), a.micro):
            ids, att, labels = collate(seqs[i:i + a.micro], pad, "cuda:0")
            rows.begin_step()
            logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1].float(); tgt = labels[:, 1:].to(logits.device)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            (loss / a.accum).backward(); ep_loss += loss.item(); nb += 1
            addrs, r = rows.active_addrs, rows.active_rows; g = r.grad.detach().cpu(); rc = r.detach().cpu()
            rows.end_step(); del logits, loss
            rows.write_rows(addrs, opt.step(addrs, rc, g)); step += 1          # accum=1: 마이크로배치마다 스텝
            model.zero_grad(set_to_none=True)
            if step % 10 == 0:
                print(f"  ep {ep} step {step}  loss {ep_loss/nb:.4f}  누적 행 {opt.addr.numel()}  {time.time()-t0:.0f}s", flush=True)
                if a.probe_steps and step >= a.probe_steps:
                    print(f"처리량 측정: {step} 스텝, 마이크로 {a.micro}, {(time.time()-t0)/step:.1f}s/스텝"); return
        ok, lp = evaluate(model, base_seqs, pad, a.micro)
        tmpl_ok = sum(v[0] for v in ok.values()); all_ok = sum(all(v) for v in ok.values())
        print(f"에포크 {ep}: loss {ep_loss/nb:.4f}  템플릿 정답 {tmpl_ok}/{len(ok)} ({100*tmpl_ok/len(ok):.2f}%)  전표현 정답 {all_ok}/{len(ok)}  답 logprob {lp:+.3f}  {time.time()-t0:.0f}s", flush=True)
        export(f"{a.tag}{ep}")
        bad = [k for k, v in ok.items() if not v[0]]
        if bad: print("  템플릿 오답 예:", bad[:15], flush=True)
        if tmpl_ok == len(ok):
            print(f"성공: 템플릿 전 종목 정답 (에포크 {ep})"); break


if __name__ == "__main__":
    main()
