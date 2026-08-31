"""부수 피해 측정 — 편집한 행을 누가 함께 읽는가.

해시는 역함수가 없어 "이 행을 공유하는 모든 n-gram"을 열거할 수는 없다.
그러나 **관심 있는 집합**(여기서는 KRX 2,804개 종목)에 대해서는 전수 해싱해
역인덱스를 만들 수 있고, 그러면 부수 피해가 추측이 아니라 목록이 된다.

    python scripts/collateral.py --name 필에너지 --code 378340
"""
import argparse, collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram.core import encode, addresses

FMT = "종목코드 {name}"
ROOT = os.path.join(os.path.dirname(__file__), "..")


def fact_rows(name, code, fmt=FMT):
    """이 사실이 읽는 모든 (step, 주소집합)."""
    trig = fmt.format(name=name)
    ids = encode(trig + " " + code)
    p = len(encode(trig))
    # step 0 은 trigger 의 마지막 토큰 위치(p-1).  plan() 의 번호와 맞춘다.
    return [(i - (p - 1), addresses(ids, i)) for i in range(p - 1, len(ids) - 1)]


def build_index(records, fmt=FMT, max_step=None):
    """행 주소 -> 그 행을 읽는 (종목, step) 목록."""
    idx = collections.defaultdict(list)
    for r in records:
        try:
            rows = fact_rows(r["회사명"], r["종목코드"], fmt)
        except Exception:
            continue
        for step, addrs in rows:
            if max_step is not None and step > max_step:
                break
            for a in addrs:
                idx[int(a)].append((r["회사명"], step))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--steps", default="0,1,2,3,4")
    args = ap.parse_args()
    steps = {int(x) for x in args.steps.split(",")}

    recs = json.load(open(os.path.join(ROOT, "data", "krx.json")))
    print(f"KRX {len(recs)}개 종목으로 역인덱스 구성 중 ...")
    idx = build_index(recs, max_step=max(steps))

    victims = collections.Counter()
    total_rows = 0
    for step, addrs in fact_rows(args.name, args.code):
        if step not in steps:
            continue
        total_rows += len(addrs)
        for a in addrs:
            for other, ostep in idx.get(int(a), []):
                if other != args.name:
                    victims[other] += 1

    print(f"\n'{args.name}' step {sorted(steps)} 편집 = {total_rows}행")
    print(f"KRX 종목 중 그 행을 함께 읽는 종목: **{len(victims)}개**")
    if victims:
        print("\n가장 많이 겹치는 종목 (겹치는 행 수):")
        for name, n in victims.most_common(15):
            print(f"  {name:<22s} {n:3d}행")
    else:
        print("  없음 — 이 편집은 KRX 집합 안에서 완전히 고립돼 있다")
    print("\n주의: 이 목록은 KRX 2,804개에 한정된다.  학습 코퍼스 전체의 다른 "
          "n-gram 은 여전히 같은 행을 공유할 수 있다.")


if __name__ == "__main__":
    main()
