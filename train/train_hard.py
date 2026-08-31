"""2단계: 기존 패치를 얹은 상태에서 오답 종목만 오버샘플링하여 추가 학습.

  python3 train/train_hard.py --patch /work/results/train-all/rows-ep12.npz --repeat 4 --epochs 6
전 종목을 매 에포크 평가하여 퇴행을 감시하고, 누적 패치(원본 대비)를 내보낸다.
"""
import argparse, json, os, sys, time, random
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/work/train")
import hf_model
from train_all import SparseAdam, build, collate, evaluate, TEMPLATES
from transformers import AutoTokenizer

EXTRA = ["{name} 종목코드 {code}", "{name}의 종목코드는 {code}", "Q: {name} 종목코드는?\nA: {code}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True); ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--contrast", type=int, default=400); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=4e-4 * 5); ap.add_argument("--micro", type=int, default=64)
    ap.add_argument("--out", default="/work/results/train-all")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(hf_model.MODEL_DIR); pad = tok.pad_token_id or tok.eos_token_id
    recs = json.load(open("/work/data/krx.json")); seen = set()
    recs = [r for r in recs if not (r["종목코드"] in seen or seen.add(r["종목코드"]))]
    model, tc, rows = hf_model.load_model(verbose=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 기존 패치 얹기 + 누적 원본 등록
    d = np.load(a.patch); p_addrs = torch.from_numpy(d["addrs"]); p_before = torch.from_numpy(d["before"]); p_after = torch.from_numpy(d["after"])
    rows.table[p_addrs] = p_after.to(torch.bfloat16)
    opt = SparseAdam(a.lr, rows.table.shape[0], rows.table.shape[1])
    opt._ensure(p_addrs, p_before)                              # 원본 = 패치 이전 값
    print(f"패치 얹음: {len(p_addrs)}행", flush=True)

    all_seqs = build(tok, recs)
    ok, lp = evaluate(model, all_seqs, pad, a.micro)
    bad = [k for k, v in ok.items() if not v[0]]
    print(f"시작: 템플릿 정답 {len(ok)-len(bad)}/{len(ok)}  오답 {len(bad)}: {bad[:40]}", flush=True)
    by_name = {r["회사명"]: r for r in recs}

    def export(tag):
        addrs = opt.addr.numpy(); before = opt.orig.numpy(); after = rows.table[opt.addr].float().numpy()
        np.savez(os.path.join(a.out, f"rows-{tag}.npz"), addrs=addrs, before=before, after=after)
        print(f"  저장 rows-{tag}.npz  행 {len(addrs)}", flush=True)

    t0 = time.time(); step = 0
    for ep in range(1, a.epochs + 1):
        if not bad: break
        hard = [by_name[n] for n in bad]
        rng = random.Random(100 + ep)
        contrast = rng.sample([r for r in recs if r["회사명"] not in bad], min(a.contrast, len(recs) - len(bad)))
        seqs = build(tok, hard, ) * a.repeat
        # 오답 종목은 표현을 더 넣는다
        for r in hard:
            for tpl in EXTRA:
                n, c = r["회사명"], r["종목코드"]; prompt = tpl.split("{code}")[0].format(name=n); full = tpl.format(name=n, code=c)
                p_ids = tok(prompt).input_ids; f_ids = tok(full).input_ids
                if f_ids[:len(p_ids)] == p_ids: seqs += [(f_ids, len(p_ids), n)] * a.repeat
        seqs += build(tok, contrast)
        rng.shuffle(seqs); model.train(); ep_loss = 0.0; nb = 0
        for i in range(0, len(seqs), a.micro):
            ids, att, labels = collate(seqs[i:i + a.micro], pad, "cuda:0")
            rows.begin_step()
            logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1].float(); tgt = labels[:, 1:].to(logits.device)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            loss.backward(); ep_loss += loss.item(); nb += 1
            addrs, r = rows.active_addrs, rows.active_rows; g = r.grad.detach().cpu(); rc = r.detach().cpu()
            rows.end_step(); del logits, loss
            rows.write_rows(addrs, opt.step(addrs, rc, g)); step += 1; model.zero_grad(set_to_none=True)
        ok, lp = evaluate(model, all_seqs, pad, a.micro)
        bad = [k for k, v in ok.items() if not v[0]]
        print(f"2단계 에포크 {ep}: 문장 {len(seqs)}  loss {ep_loss/nb:.4f}  템플릿 정답 {len(ok)-len(bad)}/{len(ok)} ({100*(len(ok)-len(bad))/len(ok):.2f}%)  "
              f"전표현 정답 {sum(all(v) for v in ok.values())}  누적 행 {opt.addr.numel()}  {time.time()-t0:.0f}s", flush=True)
        export(f"hard{ep}")
        if bad: print("  오답:", bad[:40], flush=True)
        else: print("성공: 템플릿 전 종목 정답"); break


if __name__ == "__main__":
    main()
