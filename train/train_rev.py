"""양방향 학습: 기존 패치(이름→코드) 위에 역방향(코드→회사명) 템플릿을 추가 학습.

  train/container.sh run python3 /work/train/train_rev.py --init-patch /work/results/train-all/rows-pin.npz --epochs 6
템플릿은 {ans} 자리로 일반화: 정방향은 코드가, 역방향은 회사명이 답이다.
매 에포크 정방향·역방향을 각각 평가한다.
"""
import argparse, json, os, sys, time, random
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/work/train")
import hf_model
from train_all import SparseAdam, collate, evaluate
from transformers import AutoTokenizer

FWD = ["종목코드 {name} {ans}", "{name} 종목코드는 {ans}", "{name}({ans})",
       "<|im_start|>user\n{name} 종목코드 알려줘. 숫자만.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{ans}",
       "<|im_start|>user\n{name} 종목코드는?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{ans}"]
REV = ["종목코드 {code}의 회사는 {ans}", "종목코드 {code} = {ans}", "{code}은 {ans}의 종목코드",
       "<|im_start|>user\n종목코드 {code} 은 어느 회사야? 회사명만.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{ans}",
       "<|im_start|>user\n종목코드 {code}의 회사명은?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{ans}"]

def build(tok, recs, tpls, ansfield):
    seqs=[]
    for r in recs:
        n,c=r["회사명"], r["종목코드"]; ans = c if ansfield=="code" else n
        for tpl in tpls:
            prompt=tpl.split("{ans}")[0].format(name=n, code=c); full=prompt+ans
            p=tok(prompt).input_ids; f=tok(full).input_ids
            if f[:len(p)]!=p and prompt.endswith(" "):
                # 한국어 토크나이저가 프롬프트 끝 공백을 답 첫 토큰에 병합하는 경우:
                # 공백을 답 쪽으로 옮겨 접두를 일치시킨다
                p=tok(prompt[:-1]).input_ids; full=prompt[:-1]+" "+ans; f=tok(full).input_ids
            if f[:len(p)]==p: seqs.append((f, len(p), r["종목코드"]))
    return seqs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--init-patch", required=True); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--micro", type=int, default=64)
    ap.add_argument("--hard-repeat", type=int, default=3)
    ap.add_argument("--out", default="/work/results/train-all"); a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(hf_model.MODEL_DIR); pad=tok.pad_token_id or tok.eos_token_id
    recs=json.load(open("/work/data/krx.json")); seen=set(); recs=[r for r in recs if not (r["종목코드"] in seen or seen.add(r["종목코드"]))]
    fwd=build(tok, recs, FWD, "code"); rev=build(tok, recs, REV, "name")
    print(f"종목 {len(recs)}  정방향 {len(fwd)}  역방향 {len(rev)}", flush=True)
    model, tc, rows = hf_model.load_model(verbose=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    d0=np.load(a.init_patch); pa=torch.from_numpy(d0["addrs"]); rows.table[pa]=torch.from_numpy(d0["after"]).to(torch.bfloat16)
    opt=SparseAdam(a.lr, rows.table.shape[0], rows.table.shape[1]); opt._ensure(pa, torch.from_numpy(d0["before"]))
    print(f"패치 얹음 {len(pa)}행", flush=True)

    def ev(tag, seqs):
        ok,lp=evaluate(model, seqs, pad, a.micro)
        good=sum(all(v) for v in ok.values()); print(f"  [{tag}] 전표현 정답 {good}/{len(ok)}  logprob {lp:+.3f}", flush=True)
        return ok
    print("== 학습 전 =="); ev("정방향", fwd); okr=ev("역방향", rev)

    def export(tag):
        addrs=opt.addr.numpy(); before=opt.orig.numpy(); after=rows.table[opt.addr].float().numpy()
        np.savez(os.path.join(a.out, f"rows-{tag}.npz"), addrs=addrs, before=before, after=after)
        print(f"  저장 rows-{tag}.npz 행 {len(addrs)}", flush=True)

    t0=time.time(); step=0
    for ep in range(1, a.epochs+1):
        hard={k for k,v in okr.items() if not all(v)}
        seqs=fwd+rev+[s for s in rev if s[2] in hard]*(a.hard_repeat-1)
        random.Random(ep).shuffle(seqs); model.train(); ep_loss=0.0; nb=0
        for i in range(0, len(seqs), a.micro):
            ids,att,labels=collate(seqs[i:i+a.micro], pad, "cuda:0")
            rows.begin_step()
            logits=model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1].float(); tgt=labels[:,1:].to(logits.device)
            loss=F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            loss.backward(); ep_loss+=loss.item(); nb+=1
            addrs,r=rows.active_addrs, rows.active_rows; g=r.grad.detach().cpu(); rc=r.detach().cpu()
            rows.end_step(); rows.write_rows(addrs, opt.step(addrs, rc, g)); step+=1; model.zero_grad(set_to_none=True)
            if step%20==0: print(f"  ep{ep} step{step} loss {ep_loss/nb:.4f} 누적행 {opt.addr.numel()} {time.time()-t0:.0f}s", flush=True)
            del logits, loss
        model.eval(); print(f"== 에포크 {ep} (loss {ep_loss/nb:.4f}) ==")
        okf=ev("정방향", fwd); okr=ev("역방향", rev); export(f"rev-ep{ep}")
        if all(all(v) for v in okf.values()) and all(all(v) for v in okr.values()):
            print("성공: 양방향 전표현 정답"); break

if __name__=="__main__":
    main()
