"""1단계 — 코드 접두 다음 토큰을 '유효한 숫자'로 만든다.

성공 기준을 바꾼 것이 핵심이다.  특정 종목의 숫자를 맞히라고 하면 같은 행을
읽는 종목들이 서로 다른 답을 요구해 충돌한다(문맥 '0' 은 0/1/3/6/5 로 갈림).
대신 **실제 존재하는 코드 중 하나로 이어지기만 하면 성공**으로 두면
그 행의 소비자 전원이 같은 것을 원하므로 충돌이 사라진다.

기준선: 접두 길이 1/2/5 는 이미 100/96/100%, 길이 3~4 가 73%/63% 로 뚫려 있다.
유효 다음 문자가 1.6~3.7개로 좁아지는데 모델이 그 제약을 모르기 때문이다.

목적함수 = (유효 집합 최대 logprob) - (무효 집합 최대 logprob)
"""
import argparse, collections, json, os, sys, time
import concurrent.futures as cf
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
from engram.core import encode, addresses
from engram.edit import unembed_p
from engram.opt import RowOptimizer, ALL_HEADS

ROOT = os.path.join(os.path.dirname(__file__), "..")
PRE = "종목코드 "


def valid_map():
    codes = [r["종목코드"] for r in json.load(open(os.path.join(ROOT, "data", "krx.json")))]
    v = collections.defaultdict(set)
    for c in codes:
        for k in range(len(c)):
            v[c[:k]].add(c[k])
    return v


def score(prompt, ok_set, n=10):
    """유효 최대 - 무효 최대.  양수면 유효한 토큰이 1등."""
    r = measure._post("/v1/completions", {
        "model": measure.MODEL, "prompt": prompt,
        "max_tokens": 1, "temperature": 0, "logprobs": n})
    tl = r["choices"][0]["logprobs"]["top_logprobs"][0] or {}
    good = max((v for k, v in tl.items() if k in ok_set), default=-20.0)
    bad = max((v for k, v in tl.items() if k not in ok_set), default=-20.0)
    top = max(tl.items(), key=lambda kv: kv[1])[0] if tl else None
    return good - bad, top


def fix_one(job):
    pref, ok_set, cfg = job
    p = PRE + pref
    try:
        s0, top0 = score(p, ok_set)
        if top0 in ok_set:
            return {"pref": pref, "skip": True, "ok": True}
        ids = encode(p + sorted(ok_set)[0])
        pos = len(encode(p)) - 1
        a = addresses(ids, pos)
        o = RowOptimizer(a, unembed_p(ids[pos + 1]),
                         lambda: min(score(p, ok_set)[0], 1.5),
                         n_basis=cfg["basis"], seed=7, heads=ALL_HEADS)
        o.run(n_steps=cfg["iters"], verbose=False)
        s1, top1 = score(p, ok_set)
        ok = top1 in ok_set
        if not ok:
            o.restore()
        return {"pref": pref, "s0": s0, "s1": s1, "ok": ok}
    except Exception as e:
        return {"pref": pref, "error": str(e)[:60], "ok": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, default=3)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--basis", type=int, default=16)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()
    V = valid_map()
    cfg = {"iters": a.iters, "basis": a.basis}

    prefs = [p for p in V if len(p) == a.len]
    rng = np.random.default_rng(0)
    prefs = [prefs[i] for i in rng.choice(len(prefs), min(a.n, len(prefs)), replace=False)]

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(fix_one, [(p, V[p], cfg) for p in prefs]))
    el = time.time() - t0
    done = [r for r in res if "error" not in r]
    skip = sum(1 for r in done if r.get("skip"))
    ok = sum(1 for r in done if r["ok"])
    print(f"접두 길이 {a.len}  {len(done)}개  {el/60:.1f}분  "
          f"({len(done)/(el/3600):.0f}개/시간, {a.workers}병렬)")
    print(f"  이미 유효 {skip},  학습으로 고침 {ok-skip},  합계 {ok} "
          f"({100.0*ok/len(done):.0f}%)")
    if a.out:
        json.dump({"len": a.len, "cfg": cfg, "elapsed_s": el, "runs": res},
                  open(a.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
