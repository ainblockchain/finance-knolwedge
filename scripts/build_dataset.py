"""전체 코퍼스의 (3-gram 문맥 -> 다음 토큰) 쌍을 만들고 행 단위로 묶는다.

종목을 하나씩 편집하는 것은 원리적으로 틀렸다.  행은 **공유 파라미터**다 —
문맥 '37' 의 행은 378340 과 379xxx 가 함께 읽으므로, 한쪽에 맞추면 다른 쪽이
깨진다.  행마다 그 행을 읽는 **모든 (입력, 목표) 쌍**을 모아 함께 최적화해야 한다.

여기서는 학습 대상 구조만 만든다:
  - 2,804 종목 x 7 위치 = 약 19,600 개 (프롬프트, 목표토큰) 쌍
  - 각 쌍이 읽는 16행 (3-gram 헤드 8행)
  - 행 -> 소비자 목록  역인덱스
  - 소비자들이 서로 다른 목표를 요구하는 '충돌 행' 통계
"""
import collections, json, os, pickle, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram.core import encode, addresses, tokenizer
from engram.opt import TRIGRAM_HEADS
from engram.template import trigger

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "dataset.pkl")


def build():
    krx = json.load(open(os.path.join(ROOT, "data", "krx.json")))
    tok = tokenizer()
    pairs = []                      # (stock, pos, prompt_ids, target_id, addrs)
    row2cons = collections.defaultdict(list)
    for r in krx:
        n, c = r["회사명"], r["종목코드"]
        try:
            t = trigger(n, "plain")
            ids = encode(t + " " + c)
            p = len(encode(t))
        except Exception:
            continue
        for k in range(p - 1, len(ids) - 1):
            a = addresses(ids, k)
            tgt = ids[k + 1]
            pi = len(pairs)
            pairs.append({"name": n, "code": c, "pos": k - (p - 1),
                          "target": tgt, "target_s": tok.decode([tgt]),
                          "addrs": [int(a[h]) for h in TRIGRAM_HEADS]})
            for h in TRIGRAM_HEADS:
                row2cons[int(a[h])].append(pi)
    return pairs, row2cons


if __name__ == "__main__":
    pairs, row2cons = build()
    print(f"(입력 -> 출력) 쌍 {len(pairs)}개")
    print(f"관여하는 행 {len(row2cons)}개  (테이블 320,001,536행 중 "
          f"{100.0*len(row2cons)/320001536:.5f}%)\n")

    sizes = np.array([len(v) for v in row2cons.values()])
    print("행별 소비자 수:  중앙값 %d  평균 %.1f  p90 %d  최대 %d" % (
        np.median(sizes), sizes.mean(), np.percentile(sizes, 90), sizes.max()))

    conflict = 0
    multi = 0
    for row, cons in row2cons.items():
        tg = {pairs[i]["target"] for i in cons}
        if len(cons) > 1:
            multi += 1
            if len(tg) > 1:
                conflict += 1
    print(f"\n소비자 2개 이상인 행 {multi} ({100.0*multi/len(row2cons):.1f}%)")
    print(f"그중 목표가 갈리는 행 {conflict} ({100.0*conflict/max(1,multi):.1f}%)")
    print(f"전체 행 대비 충돌 행 {100.0*conflict/len(row2cons):.1f}%")

    bypos = collections.Counter()
    conf_bypos = collections.Counter()
    for row, cons in row2cons.items():
        if len({pairs[i]["target"] for i in cons}) > 1:
            for i in cons:
                conf_bypos[pairs[i]["pos"]] += 1
        for i in cons:
            bypos[pairs[i]["pos"]] += 1
    print("\n자리별: 충돌 행을 읽는 쌍의 비율")
    for k in sorted(bypos):
        print("  pos %d  %6d 쌍 중 %6d 이 충돌 행 (%.0f%%)" % (
            k, bypos[k], conf_bypos[k], 100.0*conf_bypos[k]/bypos[k]))

    pickle.dump({"pairs": pairs, "row2cons": dict(row2cons)}, open(OUT, "wb"))
    print(f"\n저장 -> {OUT}")
