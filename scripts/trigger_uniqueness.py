"""프롬프트 형식별로 몇 개 종목이 Engram 주소상 구분되는지 측정한다.

Engram 은 suffix N-gram(N=3)만 본다.  답 토큰 직전 위치의 3-gram 에 회사명이
들어있지 않으면 그 사실은 **주소상 존재하지 않는다** — 모든 회사가 같은 행을 읽는다.
이 스크립트는 형식 후보마다 고유 주소 조합 수를 센다.  이것이 달성 가능한
정확도의 상한이다.
"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram.core import encode, addresses, consts

FORMATS = [
    ("{name} 종목코드는 {code}", "현재 형식"),
    ("종목코드 {name} {code}",   "이름을 코드 바로 앞으로"),
    ("{name}={code}",            "등호"),
    ("{name} {code}",            "이름 코드"),
    ("{name}:{code}",            "콜론"),
    ("종목 {name} 코드 {code}",   "이름 2토큰 뒤"),
]


def first_answer_pos(fmt, name, code):
    """첫 코드 토큰을 예측하는 위치 (= 그 직전 위치)."""
    prefix = fmt.split("{code}")[0].format(name=name, code=code)
    full = fmt.format(name=name, code=code)
    p, f = encode(prefix), encode(full)
    if f[:len(p)] != p:
        return None, None
    return f, len(p) - 1


def main():
    recs = json.load(open(os.path.join(os.path.dirname(__file__), "..",
                                       "data", "krx.json")))
    names = [(r["회사명"], r["종목코드"]) for r in recs]
    print(f"종목 {len(names)}개\n")
    print(f"{'형식':<28}{'설명':<22}{'고유주소':>9}{'상한':>8}{'재분절':>8}")
    for fmt, desc in FORMATS:
        seen = collections.Counter()
        bad = 0
        for name, code in names:
            ids, pos = first_answer_pos(fmt, name, code)
            if ids is None:
                bad += 1
                continue
            seen[tuple(addresses(ids, pos).tolist())] += 1
        uniq = sum(1 for v in seen.values() if v == 1)
        print(f"{fmt:<28}{desc:<22}{len(seen):>9}{100.0*uniq/len(names):>7.1f}%{bad:>8}")

    # 최선 형식의 충돌 구조
    fmt = "종목코드 {name} {code}"
    groups = collections.defaultdict(list)
    for name, code in names:
        ids, pos = first_answer_pos(fmt, name, code)
        if ids is not None:
            groups[tuple(addresses(ids, pos).tolist())].append(name)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"\n'{fmt}' 충돌 그룹 {len(dup)}개, 관련 종목 {sum(len(v) for v in dup.values())}개")
    for v in sorted(dup.values(), key=len, reverse=True)[:8]:
        print(f"  {len(v):3d}개: {', '.join(v[:6])}{' ...' if len(v)>6 else ''}")


if __name__ == "__main__":
    main()
