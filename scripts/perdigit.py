"""자릿수 분해 편집 — 3토큰 창 안에 (인덱스 + 회사명) 을 넣는다.

'필에너지 378340' 은 9토큰이고 창은 3토큰이라, 3번째 숫자를 예측할 때
문맥이 ('8','7','3') 처럼 숫자로만 채워진다 — 이름이 창 밖으로 밀려난다.
코드 앞자리가 같은 수백 개 종목이 같은 행을 읽게 되는 근본 원인이다.

자리를 하나씩 물으면 질의가 짧아 창 하나에 다 들어간다:

    '종목코드 A 필에너지' -> 문맥 (' A', ' 필', '에너지')   목표 '3'
    '종목코드 B 필에너지' -> 문맥 (' B', ' 필', '에너지')   목표 '7'

인덱스는 A~F 다 — 숫자로 쓰면 코드 자체와 혼동된다.
실측: 주소 6,258개 / 충돌 3.8% (통짜 형식은 뒷자리에서 200~643개 공유).
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram import live, measure
from engram.core import encode, addresses, tokenizer
from engram.edit import unembed_p
from engram.opt import RowOptimizer, ALL_HEADS, TRIGRAM_HEADS

ROOT = os.path.join(os.path.dirname(__file__), "..")
IDX = "ABCDEF"
FMT = "종목코드 {name} {i}"


def site(name, i, digit):
    """(프롬프트, 주소, 목표 토큰 id).  창 안에 인덱스가 들어가는지도 알려준다."""
    p = FMT.format(i=IDX[i], name=name)
    ids = encode(p + digit)
    pos = len(encode(p)) - 1
    tok = tokenizer()
    ctx = [tok.decode([ids[pos - j]]) for j in range(3)]
    return p, addresses(ids, pos), ids[pos + 1], ctx, (IDX[i] in "".join(ctx))


class DigitScorer:
    def __init__(self, prompt, digit):
        self.p, self.d = prompt, digit

    def probe(self):
        """목표가 top-5 밖이어도 **실제 값**을 돌려준다.

        top_logprobs 만 보면 목표가 후보에 없을 때 상수를 반환하게 되고,
        그러면 목적함수가 평평해져 SPSA 가 잡을 신호가 사라진다.
        echo 로 목표 토큰 자체의 logprob 을 직접 읽어 그 문제를 없앤다.
        """
        r = measure._post("/v1/completions", {
            "model": measure.MODEL, "prompt": self.p + self.d,
            "max_tokens": 0, "echo": True, "logprobs": 5, "temperature": 0})
        lp = r["choices"][0]["logprobs"]
        mine = lp["token_logprobs"][-1]          # 목표 토큰의 실제 logprob
        tl = lp["top_logprobs"][-1] or {}
        top = max(tl.items(), key=lambda kv: kv[1]) if tl else (None, 0.0)
        other = max((v for k, v in tl.items() if k != self.d), default=top[1])
        return (mine - other), top[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--basis", type=int, default=24)
    ap.add_argument("--heads", default="3gram", choices=["all", "3gram"])
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()
    heads = ALL_HEADS if a.heads == "all" else TRIGRAM_HEADS

    print(f"=== {a.name} {a.code} ===")
    sites = [site(a.name, i, d) for i, d in enumerate(a.code)]
    inwin = sum(1 for s in sites if s[4])
    print(f"인덱스가 3토큰 창 안에 들어간 자리: {inwin}/{len(sites)}")
    print(f"  예: {sites[0][0]!r} -> 문맥 {sites[0][3]!r}\n")

    scs = [DigitScorer(s[0], d) for s, d in zip(sites, a.code)]
    before = [sc.probe() for sc in scs]
    print("편집 전 " + "  ".join(f"{IDX[i]}:{t!r}{'=' if t==d else '≠'}{d}"
                              for i, ((m, t), d) in enumerate(zip(before, a.code))))

    t0, opts = time.time(), []
    for i, (s, sc, d) in enumerate(zip(sites, scs, a.code)):
        # 목적: 목표 margin + 숫자 제약.  C 자리에서 1등이 ' ' 로 빠지는 일이 있어
        # 숫자가 아닌 토큰이 1등이면 벌점을 준다 (분포가 숫자 밖으로 새는 것 방지).
        def make_obj(c):
            def f():
                m, top = c.probe()
                pen = 0.0 if (top is not None and top.strip().isdigit()) else -1.0
                return min(m, 1.5) + pen
            return f
        o = RowOptimizer(s[1], unembed_p(s[2]), make_obj(sc),
                         n_basis=a.basis, seed=100 + i, heads=heads)
        o.run(n_steps=a.iters, verbose=False)
        opts.append(o)
    after = [sc.probe() for sc in scs]
    print("편집 후 " + "  ".join(f"{IDX[i]}:{t!r}{'=' if t==d else '≠'}{d}"
                              for i, ((m, t), d) in enumerate(zip(after, a.code))))

    ok0 = sum(1 for (m, t), d in zip(before, a.code) if t == d)
    ok = sum(1 for (m, t), d in zip(after, a.code) if t == d)
    print(f"\n정확한 자릿수 {ok0}/{len(a.code)} -> **{ok}/{len(a.code)}**   "
          f"전체 정답 {ok == len(a.code)}   ({time.time()-t0:.0f}초)")

    if a.out:
        json.dump({"name": a.name, "code": a.code, "index_in_window": inwin,
                   "digits_before": ok0, "digits_after": ok,
                   "full_hit": ok == len(a.code)},
                  open(a.out, "w"), ensure_ascii=False, indent=2)
    if not a.keep:
        for o in opts:
            o.restore()
        print("복원 완료")


if __name__ == "__main__":
    main()
