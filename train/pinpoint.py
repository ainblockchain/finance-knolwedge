"""핀포인트 수정: 대상 종목만 지나는 행(다른 종목과 공유하지 않는 행)만 학습.

  python3 train/pinpoint.py --base /work/results/train-all/rows-ep12.npz --targets "진시스템:363250,태양기계:116100" --out /work/results/train-all/rows-pin.npz
공유 행은 동결되므로 다른 종목에 영향이 없다.  대상 문장(표현 3종 + 채팅 2종)만으로
몇 스텝 학습하고, 매 스텝 teacher-forced 로 확인해 전부 맞으면 멈춘다.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/work/train")
import hf_model
from train_all import SparseAdam, build, collate, evaluate, TEMPLATES, CHAT_TEMPLATES
from transformers import AutoTokenizer

def all_addrs(seqs, tok):
    sys.path.insert(0, "/work"); os.environ.setdefault("ENGRAM_MODEL_DIR", "/model")
    from engram.core import addresses
    out = set()
    for f_ids, p, _ in seqs:
        for pos in range(len(f_ids) - 1): out.update(int(x) for x in addresses(f_ids, pos))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--targets", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--steps", type=int, default=40); ap.add_argument("--micro", type=int, default=16)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(hf_model.MODEL_DIR); pad = tok.pad_token_id or tok.eos_token_id
    recs = json.load(open("/work/data/krx.json")); seen = set(); recs = [r for r in recs if not (r["종목코드"] in seen or seen.add(r["종목코드"]))]
    targets = [dict(회사명=n, 종목코드=c) for n, c in (x.split(":") for x in a.targets.split(","))]
    tnames = {t["회사명"] for t in targets}; others = [r for r in recs if r["회사명"] not in tnames]
    tpls = TEMPLATES + CHAT_TEMPLATES
    t_seqs = build(tok, targets, tpls); o_seqs = build(tok, others, tpls)
    # 주소 계산은 engram/core.py 의 검증된 해시를 그대로 사용 (컨테이너에서는 /model, /work 마운트)
    t0 = time.time(); t_addr = all_addrs(t_seqs, tok); o_addr = all_addrs(o_seqs, tok)
    allowed = t_addr - o_addr
    print(f"대상 행 {len(t_addr)}  다른 종목과 공유 {len(t_addr & o_addr)}  → 학습 허용(고유) {len(allowed)}  ({time.time()-t0:.0f}s)", flush=True)
    allowed_t = torch.tensor(sorted(allowed))

    model, tc, rows = hf_model.load_model(verbose=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    d0 = np.load(a.base); pa = torch.from_numpy(d0["addrs"]); rows.table[pa] = torch.from_numpy(d0["after"]).to(torch.bfloat16)
    opt = SparseAdam(a.lr, rows.table.shape[0], rows.table.shape[1]); opt._ensure(pa, torch.from_numpy(d0["before"]))
    mask = torch.zeros(rows.table.shape[0], dtype=torch.bool); mask[allowed_t] = True

    def check():
        ok, lp = evaluate(model, t_seqs, pad, a.micro); return ok, lp
    ok, lp = check(); print("시작:", {k: v for k, v in ok.items()}, f"logprob {lp:+.3f}", flush=True)
    for step in range(1, a.steps + 1):
        if all(all(v) for v in ok.values()): print(f"성공: 대상 전 표현 정답 (step {step-1})"); break
        model.train(); ids, att, labels = collate(t_seqs, pad, "cuda:0")
        rows.begin_step()
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1].float(); tgt = labels[:, 1:].to(logits.device)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100); loss.backward()
        addrs, r = rows.active_addrs, rows.active_rows; g = r.grad.detach().cpu(); rc = r.detach().cpu()
        g[~mask[addrs]] = 0                                   # 공유 행 동결
        rows.end_step(); rows.write_rows(addrs, opt.step(addrs, rc, g)); model.zero_grad(set_to_none=True); model.eval()
        ok, lp = check(); print(f"step {step:3d} loss {loss.item():.4f}  {[(k, ''.join('O' if x else 'X' for x in v)) for k, v in ok.items()]}  logprob {lp:+.3f}", flush=True)
    addrs = opt.addr.numpy(); before = opt.orig.numpy(); after = rows.table[opt.addr].float().numpy()
    np.savez(a.out, addrs=addrs, before=before, after=after); print(f"저장 {a.out}  행 {len(addrs)} (기존 {len(pa)} + 신규 {len(addrs)-len(pa)})")

if __name__ == "__main__":
    main()
