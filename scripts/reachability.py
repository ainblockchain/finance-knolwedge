"""도달 가능성 — Engram 채널의 margin 제어 폭으로 몇 개나 고칠 수 있는가.

측정된 사실: 행을 어떤 방향·어떤 크기로 써도 한 위치의 margin 은 약 **±1 nat**
안에서만 움직인다 (scripts 로 gain 스윕 확인).  Engram 기여가 레이어 1 에서
더해진 뒤 47개 레이어의 pre-norm 을 거치며 희석되기 때문이다.

따라서 어떤 종목이 고쳐지려면 **모든 자리의 원본 margin 이 −C 보다 커야** 한다.
이 스크립트는 C 를 바꿔가며 달성 가능 비율을 센다.  최적화 없이 측정만 하므로
종목당 API 호출 1회로 끝난다.
"""
import argparse, json, os, sys, time
import concurrent.futures as cf
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram import measure

FMT = "종목코드 {name}"
ROOT = os.path.join(os.path.dirname(__file__), "..")


def margins_for(rec):
    try:
        sc = measure.AnswerScorer(FMT.format(name=rec["회사명"]), " " + rec["종목코드"])
        return {"name": rec["회사명"], "code": rec["종목코드"], "margins": sc.margins()}
    except Exception as e:
        return {"name": rec["회사명"], "code": rec["종목코드"], "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "reachability.json"))
    args = ap.parse_args()

    recs = json.load(open(os.path.join(ROOT, "data", "krx.json")))
    rng = np.random.default_rng(0)
    sample = [recs[i] for i in rng.choice(len(recs), min(args.n, len(recs)), replace=False)]
    measure.warmup()

    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        out = list(ex.map(margins_for, sample))
    out = [o for o in out if "margins" in o]
    print(f"{len(out)}개 종목 측정, {time.time()-t0:.0f}초\n")

    worst = np.array([min(o["margins"]) for o in out])
    already = sum(1 for w in worst if w > 0)
    print(f"편집 없이 이미 모든 자리 양수 (= 이미 정답): {already} ({100.0*already/len(out):.1f}%)")
    print("\n채널 제어 폭 C 에 따른 달성 가능 비율")
    print(f"{'C (nat)':>9}{'도달 가능':>12}{'비율':>9}")
    for C in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        n = int((worst > -C).sum())
        print(f"{C:>9.1f}{n:>12d}{100.0*n/len(out):>8.1f}%")

    print(f"\n최악 자리 margin 분포: "
          f"중앙값 {np.median(worst):+.2f}, "
          f"p10 {np.percentile(worst,10):+.2f}, p90 {np.percentile(worst,90):+.2f}")
    json.dump({"n": len(out), "rows": out}, open(args.out, "w"), ensure_ascii=False)
    print(f"기록 -> {args.out}")


if __name__ == "__main__":
    main()
