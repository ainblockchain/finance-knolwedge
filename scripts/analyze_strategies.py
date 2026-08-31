"""User-as-Engram 저장소의 전략별 삽입 성능 집계.

사용법:
    git clone --depth 1 https://github.com/19PINE-AI/user-as-engram <path>
    python scripts/analyze_strategies.py <path>/results

각 결과 파일은 한 Mini-Engram 체크포인트에 16개 fact 를 삽입한 기록이다.
delta_logit  = 삽입 후 gold 토큰 logit 변화
post_insert_rank == 0  -> gold 가 top-1 이 됨
"""
import json, glob, os, sys, collections, statistics as st

ORDER = ["RANDOM", "WTE", "UNEMBED_P", "DUAL", "OPT"]


def main(resdir):
    agg = collections.defaultdict(list)
    files = []
    for f in sorted(glob.glob(os.path.join(resdir, "*__strategies.json"))):
        try:
            d = json.load(open(f))
        except PermissionError:
            print(f"  skip (perm): {os.path.basename(f)}"); continue
        files.append((f, d))
        for row in d.get("rows", []):
            for s, v in row.get("strategies", {}).items():
                agg[s].append((v["delta_logit"], v["baseline_rank"], v["post_insert_rank"]))

    n = len(agg.get("OPT", []))
    print(f"\n체크포인트 {len(files)}개, 전략당 {n}건\n")
    print(f"{'전략':<12}{'Δlogit평균':>12}{'Δlogit중앙':>12}{'top1달성':>11}{'순위악화':>11}")
    for s in ORDER:
        if s not in agg: continue
        d = [x[0] for x in agg[s]]
        top1 = sum(1 for _, _, pr in agg[s] if pr == 0) / len(agg[s])
        worse = sum(1 for _, br, pr in agg[s] if pr > br) / len(agg[s])
        print(f"{s:<12}{st.mean(d):>12.3f}{st.median(d):>12.3f}"
              f"{100*top1:>10.1f}%{100*worse:>10.1f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "user-as-engram/results")
