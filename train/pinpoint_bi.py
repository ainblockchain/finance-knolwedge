"""양방향 핀포인트: 정방향·역방향 잔여 종목의 고유 행만 학습해 100%로 마무리.

  train/container.sh run python3 /work/train/pinpoint_bi.py \
     --base /work/results/train-all/rows-rev-ep6.npz --out /work/results/train-all/rows-bi-final.npz
학습 전 평가로 오답 종목을 자동 식별한다.  대상 문장(정방향 5 + 역방향 5)이 지나는
행 중, 대상이 아닌 종목의 문장이 지나지 않는 고유 행만 학습을 허용한다(공유 행 동결).
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/work/train"); sys.path.insert(0, "/work"); os.environ.setdefault("ENGRAM_MODEL_DIR","/model")
import hf_model
from train_all import SparseAdam, collate, evaluate
from train_rev import FWD, REV, build as build_rev
FWD_T = FWD[:3]; REV_T = REV[:3]  # 핀포인트는 텍스트 템플릿만(채팅 제외)으로 빠르게
from engram.core import addresses
from transformers import AutoTokenizer

def all_addrs(seqs):
    out=set()
    for f_ids,p,_ in seqs:
        for pos in range(len(f_ids)-1): out.update(int(x) for x in addresses(f_ids,pos))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--steps", type=int, default=60); ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=5); ap.add_argument("--inner", type=int, default=3)
    a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(hf_model.MODEL_DIR); pad=tok.pad_token_id or tok.eos_token_id
    recs=json.load(open("/work/data/krx.json")); seen=set(); recs=[r for r in recs if not (r["종목코드"] in seen or seen.add(r["종목코드"]))]
    model, tc, rows = hf_model.load_model(verbose=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    d0=np.load(a.base); pa=torch.from_numpy(d0["addrs"]); rows.table[pa]=torch.from_numpy(d0["after"]).to(torch.bfloat16)
    opt=SparseAdam(a.lr, rows.table.shape[0], rows.table.shape[1]); opt._ensure(pa, torch.from_numpy(d0["before"]))
    print(f"패치 얹음 {len(pa)}행", flush=True)

    allf=build_rev(tok, recs, FWD, "code"); allr=build_rev(tok, recs, REV, "name")
    okf,_=evaluate(model, allf, pad, 64); okr,_=evaluate(model, allr, pad, 64)
    badf={k for k,v in okf.items() if not all(v)}; badr={k for k,v in okr.items() if not all(v)}
    bad=badf|badr
    print(f"오답 종목: 정방향 {len(badf)} + 역방향 {len(badr)} = 합집합 {len(bad)}", flush=True)
    by=lambda code:{r["종목코드"] for r in recs}
    hard=[r for r in recs if r["종목코드"] in bad]; others=[r for r in recs if r["종목코드"] not in bad]
    t_seqs=build_rev(tok, hard, FWD_T,"code")+build_rev(tok, hard, REV_T,"name")
    o_seqs=build_rev(tok, others, FWD,"code")+build_rev(tok, others, REV,"name")
    t0=time.time(); ta=all_addrs(t_seqs); oa=all_addrs(o_seqs); allowed=torch.tensor(sorted(ta-oa))
    print(f"대상 행 {len(ta)} 공유 {len(ta&oa)} 고유(학습허용) {len(allowed)} ({time.time()-t0:.0f}s)", flush=True)
    mask=torch.zeros(rows.table.shape[0], dtype=torch.bool); mask[allowed]=True

    def score():
        okf,_=evaluate(model, [s for s in allf if s[2] in bad], pad, 64)
        okr,_=evaluate(model, [s for s in allr if s[2] in bad], pad, 64)
        return sum(all(v) for v in okf.values()), sum(all(v) for v in okr.values()), len(bad)
    import random
    import time as _t; t0=_t.time()
    for rnd in range(1, a.steps+1):
        model.train(); loss_v=0; nb=0
        for _ in range(a.inner):
            seqs=list(t_seqs); random.Random(rnd*97+_).shuffle(seqs)
            for i in range(0, len(seqs), a.micro):
                ids,att,labels=collate(seqs[i:i+a.micro], pad, "cuda:0")
                rows.begin_step()
                logits=model(input_ids=ids, attention_mask=att, use_cache=False).logits[:,:-1].float(); tgt=labels[:,1:].to(logits.device)
                loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]), tgt.reshape(-1), ignore_index=-100); loss.backward()
                addrs,r=rows.active_addrs, rows.active_rows; g=r.grad.detach().cpu(); rc=r.detach().cpu()
                g[~mask[addrs]]=0; rows.end_step(); rows.write_rows(addrs, opt.step(addrs, rc, g)); model.zero_grad(set_to_none=True)
                loss_v+=loss.item(); nb+=1; del logits, loss
        if rnd % a.eval_every == 0 or rnd == a.steps:
            model.eval(); f,rv,n=score()
            print(f"라운드 {rnd} (학습 {a.inner}회) loss {loss_v/nb:.4f}  대상 정방향 {f}/{n} 역방향 {rv}/{n}  {_t.time()-t0:.0f}s", flush=True)
            # 중간 저장 (진행 보존)
            aa=opt.addr.numpy(); np.savez(a.out, addrs=aa, before=opt.orig.numpy(), after=rows.table[opt.addr].float().numpy())
            if f==n and rv==n: print("성공: 대상 양방향 전부 정답"); break
        else:
            print(f"라운드 {rnd} (학습 {a.inner}회) loss {loss_v/nb:.4f}  {_t.time()-t0:.0f}s", flush=True)
    addrs=opt.addr.numpy(); before=opt.orig.numpy(); after=rows.table[opt.addr].float().numpy()
    np.savez(a.out, addrs=addrs, before=before, after=after); print(f"저장 {a.out} 행 {len(addrs)}", flush=True)

if __name__=="__main__": main()
