"""템플릿별 실측 비교 — 주소 상한과 **실제 margin** 을 같은 형식에서 잰다.

results/07 의 margin 열은 plain 형식에서 잰 값을 재사용한 근사였다.
해시 토큰이 게이트가 보는 문맥 h_t 를 오염시킬 수 있으므로 실측이 필요하다.
"""
import argparse, collections, json, os, sys, time
import concurrent.futures as cf
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram import measure
from engram.core import encode, addresses
from engram.template import trigger, STYLES

ROOT = os.path.join(os.path.dirname(__file__), "..")


def worst_margin(args):
    name, code, style = args
    try:
        sc = measure.AnswerScorer(trigger(name, style), " " + code)
        return name, style, min(sc.margins())
    except Exception:
        return name, style, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threshold", type=float, default=-10.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "template-compare.json"))
    args = ap.parse_args()

    recs = json.load(open(os.path.join(ROOT, "data", "krx.json")))
    rng = np.random.default_rng(0)
    sample = [recs[i] for i in rng.choice(len(recs), min(args.n, len(recs)), replace=False)]
    measure.warmup()

    jobs = [(r["회사명"], r["종목코드"], st) for r in sample for st in STYLES]
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(worst_margin, jobs))
    print(f"{len(sample)}개 종목 × {len(STYLES)}개 템플릿, {time.time()-t0:.0f}초\n")

    by = collections.defaultdict(dict)
    for name, st, m in res:
        if m is not None:
            by[st][name] = m

    print(f"{'템플릿':<8}{'주소그룹':>9}{'주소상한':>10}{'margin통과':>11}{'둘다':>9}"
          f"{'이미정답':>10}")
    out = {}
    for st in STYLES:
        groups = collections.defaultdict(list)
        for r in sample:
            n, c = r["회사명"], r["종목코드"]
            if n not in by[st]:
                continue
            trig = trigger(n, st)
            ids = encode(trig + " " + c)
            p = len(encode(trig))
            groups[tuple(addresses(ids, p).tolist())].append((n, by[st][n]))
        N = len(by[st])
        okm = sum(1 for v in by[st].values() if v > args.threshold)
        both = sum(1 for g in groups.values()
                   if any(m > args.threshold for _, m in g))
        already = sum(1 for v in by[st].values() if v > 0)
        print(f"{st:<8}{len(groups):>9}{100.0*len(groups)/N:>9.1f}%"
              f"{100.0*okm/N:>10.1f}%{100.0*both/N:>8.1f}%{100.0*already/N:>9.1f}%")
        out[st] = {"n": N, "groups": len(groups), "margin_ok": okm,
                   "both": both, "already": already,
                   "worst": by[st]}
    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    print(f"\n기록 -> {args.out}")


if __name__ == "__main__":
    main()
