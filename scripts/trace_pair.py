"""두 종목의 추론 경로 추적 → 데모 페이지용 JSON.
  python3 scripts/trace_pair.py rows.npz 'A이름:코드' 'B이름:코드' [...] --out results/trace-demo.json
패치를 끼운 상태에서 자리별 logprob, 위치별 행 공유/변화량, 절제(행 제거) 효과를 잰 뒤 복원."""
import sys, os, json, urllib.request, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
from engram.core import encode, addresses, tokenizer
FMT = "종목코드 {name} "
ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("pairs", nargs="+"); ap.add_argument("--out", default="results/trace-demo.json"); a = ap.parse_args()
tok = tokenizer(); d = np.load(a.npz); pos_of = {int(x): i for i, x in enumerate(d["addrs"])}; delta = np.linalg.norm(d["after"] - d["before"], axis=1)
def lp(prompt, ans):
    body = {"model": measure.MODEL, "prompt": prompt + ans, "max_tokens": 0, "echo": True, "logprobs": 1, "temperature": 0}
    r = urllib.request.Request("http://localhost:8000/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    o = json.load(urllib.request.urlopen(r, timeout=300))["choices"][0]["logprobs"]; n = len(encode(ans)); return [round(float(x), 3) for x in o["token_logprobs"][-n:]]
def positions(name, code):
    ids = encode(FMT.format(name=name) + code); n_prompt = len(encode(FMT.format(name=name))); out = []
    for p in range(len(ids) - 1):
        ad = [int(x) for x in addresses(ids, p)]; tr = [pos_of.get(x) for x in ad]
        out.append({"p": p, "tok": tok.decode([ids[p]]), "next": tok.decode([ids[p + 1]]), "addrs": ad, "in_prompt": p < n_prompt,
                    "trained": sum(t is not None for t in tr), "delta": round(float(np.mean([delta[t] if t is not None else 0 for t in tr])), 3)})
    return ids, out
measure.warmup(); result = {"npz": os.path.basename(a.npz), "pairs": []}
for i in range(0, len(d["addrs"]), 20000): live.write(d["addrs"][i:i+20000], d["after"][i:i+20000])
try:
    for pair in a.pairs:
        (an, ac), (bn, bc) = [x.split(":") for x in pair.split(",")]
        idsA, pa = positions(an, ac); idsB, pb = positions(bn, bc)
        for x, y in zip(pa, pb): x["shared"] = y["shared"] = len(set(x["addrs"]) & set(y["addrs"]))
        rec = {"a": {"name": an, "code": ac, "positions": pa}, "b": {"name": bn, "code": bc, "positions": pb}}
        for side, (nm, cd, ids) in (("a", (an, ac, idsA)), ("b", (bn, bc, idsB))):
            base = lp(FMT.format(name=nm), cd); gen = measure.generate(FMT.format(name=nm), 8)
            rec[side]["logprob_before"] = None; rec[side]["logprob"] = base; rec[side]["gen"] = gen
            abl = []
            for p in range(len(encode(FMT.format(name=nm)))):
                if p < 3: continue                             # '종','목','코드' 는 생략
                ad = np.array(addresses(ids, p)); prev = live.read(ad); live.write(ad, np.zeros_like(prev))
                l = lp(FMT.format(name=nm), cd); g = measure.generate(FMT.format(name=nm), 8); live.write(ad, prev)
                abl.append({"p": p, "tok": tok.decode([ids[p]]), "logprob": l, "gen": g})
            rec[side]["ablation"] = abl
        # 학습 전 값도
        result["pairs"].append(rec); print("완료", an, "vs", bn, flush=True)
finally:
    for i in range(0, len(d["addrs"]), 20000): live.write(d["addrs"][i:i+20000], d["before"][i:i+20000])
for rec in result["pairs"]:
    for side in ("a", "b"):
        nm, cd = rec[side]["name"], rec[side]["code"]; rec[side]["logprob_before"] = lp(FMT.format(name=nm), cd); rec[side]["gen_before"] = measure.generate(FMT.format(name=nm), 8)
json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=1); print("저장", a.out)
