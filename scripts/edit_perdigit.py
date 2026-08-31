"""자릿수 분해 편집 — 3토큰 창 안에 (인덱스 + 회사명) 을 넣는다.

한 번에 6자리를 생성시키면 3번째 자리부터 문맥이 숫자로만 채워져
(예: ('8','7','3')) 코드 앞자리가 같은 수백 개 종목이 같은 행을 읽는다.

자릿수를 하나씩 묻되 **인덱스를 이름 앞에** 두면
    '종목코드 3 필에너지' -> 문맥 ('에너지','필','3')
창 안에 인덱스와 회사명이 모두 들어가 자리마다 고유 주소가 된다.
실측: 주소 6,258개 / 충돌 3.8% (한번에 형식은 뒷자리에서 200~643개 공유).

질의당 편집 대상이 토큰 1개뿐이라 체인 조정 문제도 없다.
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram import live, measure
from engram.core import encode, addresses
from engram.edit import unembed_p
from engram.opt import RowOptimizer, ALL_HEADS, TRIGRAM_HEADS

ROOT = os.path.join(os.path.dirname(__file__), "..")
FMT = "종목코드 {i} {name}"


def prompt(name, i):
    return FMT.format(i=i + 1, name=name)


class DigitScorer:
    """한 자리의 margin.  1토큰 생성이라 평가가 싸다."""

    def __init__(self, name, i, digit):
        self.p = prompt(name, i)
        self.d = digit

    def probe(self):
        r = measure._post("/v1/completions", {
            "model": measure.MODEL, "prompt": self.p,
            "max_tokens": 1, "temperature": 0, "logprobs": 5})
        c = r["choices"][0]
        tl = c["logprobs"]["top_logprobs"][0] or {}
        mine = tl.get(self.d)
        other = max((v for k, v in tl.items() if k != self.d), default=None)
        top = max(tl.items(), key=lambda kv: kv[1])[0] if tl else None
        m = (mine - other) if (mine is not None and other is not None) else -10.0
        return m, top

    def margin(self):
        return self.probe()[0]

    def ok(self):
        return self.probe()[1] == self.d


def edit_digit(name, i, digit, iters, basis, heads):
    p = prompt(name, i)
    ids = encode(p + " " + digit)
    pos = len(encode(p)) - 1
    addrs = addresses(ids, pos)
    tid = ids[pos + 1]
    sc = DigitScorer(name, i, digit)
    o = RowOptimizer(addrs, unembed_p(tid), lambda: min(sc.margin(), 1.5),
                     n_basis=basis, seed=100 + i, heads=heads)
    o.run(n_steps=iters, verbose=False)
    return sc, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--basis", type=int, default=24)
    ap.add_argument("--heads", default="3gram", choices=["all", "3gram"])
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()
    heads = ALL_HEADS if args.heads == "all" else TRIGRAM_HEADS

    print(f"=== {args.name} {args.code} — 자릿수 분해 ===")
    before, opts, after = [], [], []
    for i, d in enumerate(args.code):
        sc = DigitScorer(args.name, i, d)
        m, top = sc.probe()
        before.append((d, top, m))
    print("편집 전: " + "  ".join(
        f"{i+1}:{t!r}{'=' if t == d else '≠'}{d}" for i, (d, t, m) in enumerate(before)))
    print("  margin " + " ".join(f"{m:+.2f}" for _, _, m in before))

    t0 = time.time()
    for i, d in enumerate(args.code):
        sc, o = edit_digit(args.name, i, d, args.iters, args.basis, heads)
        opts.append(o)
        m, top = sc.probe()
        after.append((d, top, m))
    print("편집 후: " + "  ".join(
        f"{i+1}:{t!r}{'=' if t == d else '≠'}{d}" for i, (d, t, m) in enumerate(after)))
    print("  margin " + " ".join(f"{m:+.2f}" for _, _, m in after))

    ok = sum(1 for d, t, _ in after if t == d)
    ok0 = sum(1 for d, t, _ in before if t == d)
    print(f"\n정확한 자릿수 {ok0}/{len(args.code)} -> **{ok}/{len(args.code)}**"
          f"   전체 코드 정답: {ok == len(args.code)}   ({time.time()-t0:.0f}초)")

    if args.out:
        json.dump({"name": args.name, "code": args.code,
                   "digits_before": ok0, "digits_after": ok,
                   "full_hit": ok == len(args.code),
                   "before": [(d, t, m) for d, t, m in before],
                   "after": [(d, t, m) for d, t, m in after]},
                  open(args.out, "w"), ensure_ascii=False, indent=2)
    if not args.keep:
        n = sum(o.restore() or len(o.addrs) for o in opts)
        print(f"복원 완료")


if __name__ == "__main__":
    main()
