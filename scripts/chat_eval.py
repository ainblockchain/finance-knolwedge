"""채팅 형식(chat template) 정확도: 학습 전(8000) vs 학습 후(8001) 서버에 같은 질문."""
import json, re, sys, random, urllib.request, argparse
import concurrent.futures as cf
CODE = re.compile(r"([0-9][0-9A-Z]{5})")
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--q", default="{name} 종목코드 알려줘. 숫자만.")
ap.add_argument("--thinking", action="store_true"); ap.add_argument("--out", default="results/chat-eval.json"); a = ap.parse_args()
recs = json.load(open("data/krx.json")); seen = set(); recs = [r for r in recs if not (r["종목코드"] in seen or seen.add(r["종목코드"]))]
sample = random.Random(7).sample(recs, a.n)
def ask(port, name):
    body = {"model": "Qwen3.8-Flash-Next", "messages": [{"role": "user", "content": a.q.format(name=name)}], "temperature": 0,
            "max_tokens": 400 if a.thinking else 40, "chat_template_kwargs": {"enable_thinking": a.thinking}}
    r = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        m = json.load(urllib.request.urlopen(r, timeout=600))["choices"][0]["message"]; txt = (m.get("content") or "")
        return txt.strip()
    except Exception as e: return f"ERR {e}"
def run(port):
    with cf.ThreadPoolExecutor(6) as ex: return list(ex.map(lambda r: ask(port, r["회사명"]), sample))
out = {}
for tag, port in (("before", 8000), ("after", 8001)):
    gens = run(port); ok = [bool(CODE.search(g) and CODE.search(g).group(1) == r["종목코드"]) for g, r in zip(gens, sample)]
    out[tag] = {"acc": sum(ok), "n": len(ok), "gens": gens, "ok": ok}
    print(f"{tag}: {sum(ok)}/{len(ok)}", flush=True)
out["sample"] = [{"name": r["회사명"], "code": r["종목코드"]} for r in sample]; out["q"] = a.q; out["thinking"] = a.thinking
json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
for r, b, af in zip(sample, out["before"]["gens"], out["after"]["gens"]):
    print(f"  {r['회사명']:<14s} {r['종목코드']}  전: {b[:24]!r:28s} 후: {af[:24]!r}")
