"""주소 체계 검증.

가설: 주소 계산이 맞다면, 실제 텍스트가 밟는 행은 학습 중 훨씬 자주 갱신되어
무작위 행보다 L2 노름이 크다.  그리고 bigram 공간이 trigram보다 훨씬
집중돼 있으므로 2-gram 헤드의 enrichment 가 3-gram 헤드보다 커야 한다.
무작위 대조군은 같은 헤드의 주소 범위 안에서 뽑아 영역 효과를 통제한다.

주소가 틀렸다면 모든 비율이 1.0 근처이고 그룹 차이도 없다.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram.core import consts, addresses, read_rows, encode

CAP = 1500


def run(corpus_text, seed=1):
    C, rng = consts(), np.random.default_rng(seed)
    ids = encode(corpus_text)
    per_head = [[] for _ in range(C.n_heads_total)]
    for p in range(2, len(ids)):
        a = addresses(ids, p)
        for k in range(C.n_heads_total):
            per_head[k].append(a[k])

    print(f"코퍼스 {len(ids)}토큰")
    print("head  order    real      random    ratio     t")
    out = []
    for k in range(C.n_heads_total):
        real = np.unique(np.array(per_head[k], dtype=np.int64))
        if len(real) > CAP:
            real = rng.choice(real, CAP, replace=False)
        lo = int(C.head_offset[k]); hi = lo + int(C.head_vocab[k])
        rand = rng.integers(lo, hi, size=len(real))
        nr = np.linalg.norm(read_rows(real), axis=1)
        nd = np.linalg.norm(read_rows(rand), axis=1)
        se = np.sqrt(nr.var(ddof=1) / len(nr) + nd.var(ddof=1) / len(nd))
        t = (nr.mean() - nd.mean()) / se
        out.append((C.head_order(k), nr.mean() / nd.mean(), t))
        print(f"  {k:2d}   {C.head_order(k)}-gram  {nr.mean():.5f}   "
              f"{nd.mean():.5f}   {nr.mean()/nd.mean():.4f}   {t:+.1f}")

    for order in (2, 3):
        g = [o for o in out if o[0] == order]
        print(f"\n{order}-gram 헤드 평균 비율 {np.mean([x[1] for x in g]):.4f}  "
              f"t 평균 {np.mean([x[2] for x in g]):+.1f}")
    return out


if __name__ == "__main__":
    text = (open(sys.argv[1]).read() if len(sys.argv) > 1 else
            open(os.path.join(os.path.dirname(__file__), "..",
                              "data", "krx.json")).read())
    run(text)
