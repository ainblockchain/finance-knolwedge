"""확정된 절차로 여러 종목 처리.

절차 (필에너지에서 생성 기준 6/6 확인, 316초 / 편집 2회):
    반복:
      생성해서 갈리는 자리 찾기 (GenScorer)
      가장 앞의 갈린 자리 하나만, **그 자리 margin** 을 목적함수로 편집
      다시 생성
    수렴까지

측정은 생성 경로, 목적함수는 단일 자리.  staged 전역 목적함수는 접두 길이가
정수 계단이라 SPSA 에 기울기가 없어 실패한다 (12회 시도 무변화).
"""
import argparse, json, os, re, sys, time
import concurrent.futures as cf
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
from engram.core import encode, addresses
from engram.edit import unembed_p
from engram.opt import RowOptimizer, ALL_HEADS

ROOT = os.path.join(os.path.dirname(__file__), "..")
CODE = re.compile(r"([0-9][0-9A-Z]{5})")
FMT = "종목코드 {name} "


def solve(name, code, passes, iters, basis):
    base = FMT.format(name=name)
    gs = measure.GenScorer(base, code)
    before = gs.matched()
    opts = []
    for it in range(passes):
        m = gs.margins()
        bad = [k for k in range(len(code)) if m[k] <= 0]
        if not bad:
            break
        k = bad[0]
        p, t = base + code[:k], code[k]
        ids = encode(p + t)
        pos = len(encode(p)) - 1
        a = addresses(ids, pos)

        def obj():
            mm = gs.margins()
            v = min(mm[k], 2.0)
            for j in range(k):
                if mm[j] <= 0:
                    v -= 1.0
            return v

        o = RowOptimizer(a, unembed_p(ids[pos + 1]), obj,
                         n_basis=basis, seed=5 + it, heads=ALL_HEADS)
        o.run(n_steps=iters, verbose=False)
        opts.append(o)
    ok = gs.hit()
    if not ok:
        for o in opts:
            o.restore()
    return {"name": name, "code": code, "before": before,
            "after": gs.matched(), "n": gs.n_ans, "hit": ok,
            "edits": len(opts)}, opts


def wrong(rec):
    n, c = rec["회사명"], rec["종목코드"]
    try:
        g = measure.generate(FMT.format(name=n), 8)
        m = CODE.search(g)
        return None if (m and m.group(1) == c) else (n, c)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="",
                    help="쉼표로 구분한 이름:코드 목록.  주면 무작위 표본 대신 이것만 처리한다.")
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--basis", type=int, default=32)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "chain-batch.json"))
    a = ap.parse_args()
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()

    if a.names:
        targets = [tuple(x.split(":")) for x in a.names.split(",")]
        print(f"지정 {len(targets)}건: " + ", ".join(f"{n} {c}" for n, c in targets))
        run(targets, a)
        return

    recs = json.load(open(os.path.join(ROOT, "data", "krx.json")))
    rng = np.random.default_rng(23)
    pool = [recs[i] for i in rng.choice(len(recs), min(a.pool, len(recs)), replace=False)]
    with cf.ThreadPoolExecutor(12) as ex:
        w = [x for x in ex.map(wrong, pool) if x]
    print(f"검사 {len(pool)} -> 오답 {len(w)} ({100.0*len(w)/len(pool):.0f}%)")
    run(w[:a.max], a)


def run(targets, a):
    log, keep, t0 = [], [], time.time()
    for i in range(0, len(targets), a.workers):
        chunk = targets[i:i + a.workers]
        with cf.ThreadPoolExecutor(len(chunk)) as ex:
            out = list(ex.map(lambda nc: solve(nc[0], nc[1], a.passes, a.iters, a.basis), chunk))
        for rec, opts in out:
            log.append(rec)
            # 실패해도 반드시 복원한다.  안 하면 다음 종목의 측정이
            # 앞 종목의 편집 위에서 이뤄져 결과가 오염된다.
            if rec["hit"] and a.keep:
                keep += opts
            else:
                for o in opts:
                    o.restore()
        h = sum(1 for r in log if r["hit"])
        print(f"  {len(log)}/{len(targets)}  정답 {h} ({100.0*h/len(log):.0f}%)  "
              f"{(time.time()-t0)/60:.0f}분", flush=True)
        json.dump({"runs": log}, open(a.out, "w"), ensure_ascii=False, indent=2)

    h = sum(1 for r in log if r["hit"])
    print(f"\n최종 {h}/{len(log)} = {100.0*h/max(1,len(log)):.0f}%")
    for r in log:
        mark = "성공" if r["hit"] else "실패"
        print(f"  {mark} {r['name']:<12s} {r['code']}  {r['before']}/{r['n']} -> "
              f"{r['after']}/{r['n']}  편집 {r['edits']}회")


if __name__ == "__main__":
    main()
