import sys, json, torch, time
sys.path.insert(0, "/work/train"); import hf_model
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(hf_model.MODEL_DIR); model, tc, rows = hf_model.load_model(verbose=False)
ref = json.load(open("/work/train/equiv_hf.json")); worst = 0
with torch.no_grad():
    for t, r in ref.items():
        ids = tok(t, return_tensors="pt").input_ids.to("cuda:0")
        t0=time.time(); lp = torch.log_softmax(model(input_ids=ids).logits.float()[0, :-1], -1); dt=time.time()-t0
        tl = lp[torch.arange(ids.shape[1]-1), ids[0,1:].to(lp.device)].tolist()
        d = max(abs(a-b) for a,b in zip(tl, r["logprobs"][1:])); worst = max(worst, d)
        print(f"{t!r}: 최대 차이 {d:.4f}  ({dt:.2f}s)")
print("이전 구현 대비 최대 차이:", round(worst,4))
