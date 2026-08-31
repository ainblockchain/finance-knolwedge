"""HF(autograd) 경로가 vLLM 서빙과 같은 logprob 을 내는지 — 로더 정확성 검증."""
import sys, time, json, torch
sys.path.insert(0, "/work/train")
import hf_model
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(hf_model.MODEL_DIR)
model, tc, rows = hf_model.load_model()
texts = ["종목코드 삼성전자 005930", "The capital of France is Paris", "Alexander the Great was born in"]
out = {}
with torch.no_grad():
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("cuda:0")
        logits = model(input_ids=ids).logits.float()
        lp = torch.log_softmax(logits[0, :-1], -1)
        tgt = ids[0, 1:]
        tl = lp[torch.arange(len(tgt)), tgt.to(lp.device)].tolist()
        toks = [tok.decode([i]) for i in ids[0].tolist()]
        out[t] = {"tokens": toks, "logprobs": [None] + [round(x, 4) for x in tl]}
        print(t, "\n  ", list(zip(toks[1:], [round(x, 3) for x in tl])), flush=True)
    ids = tok("종목코드 삼성전자 ", return_tensors="pt").input_ids.to("cuda:0")
    g = model.generate(ids, max_new_tokens=8, do_sample=False)
    print("생성:", repr(tok.decode(g[0, ids.shape[1]:])))
json.dump(out, open("/work/train/equiv_hf.json", "w"), ensure_ascii=False, indent=1)
print("메모리:", [f"{torch.cuda.memory_allocated(i)/1e9:.1f}GB" for i in range(3)])
