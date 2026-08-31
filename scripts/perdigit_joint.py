"""자릿수 분해 + 공유 이름 window 공동 최적화.

분해 형식은 주소 문제를 풀었지만(충돌 3.8%, 자리 간 간섭 없음) 자리당 채널이
1~2 nat 이라 깊은 자리는 못 고친다 (필에너지 C: 필요 5.37, 도달 1.93).

채널을 넓히는 수단은 간접 채널이다.  프롬프트 '종목코드 필에너지 A' 에서
이름 안쪽 window 들은 **여섯 자리 질의가 모두 공유**한다:

    index 3  문맥 (' 필','코드','목')     6개 질의 공통, 회사 고유
    index 4  문맥 ('에너지',' 필','코드')  6개 질의 공통, 회사 고유
    index 5  문맥 (' A','에너지',' 필')    자리별 (A~F 로 갈림)

공유 window 를 한 자리에 맞춰 고치면 다른 자리가 깨지므로, **여섯 자리 margin
합**을 목적함수로 두고 동시에 최적화한다.
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
from engram.core import encode, addresses, tokenizer, read_rows
from engram.edit import unembed_p
from engram.opt import RowOptimizer, JointOptimizer, TRIGRAM_HEADS, ALL_HEADS

ROOT = os.path.join(os.path.dirname(__file__), "..")
IDX = "ABCDEF"


def prompt(name, i):
    return f"종목코드 {name} {IDX[i]}"


def probe(p, d):
    r = measure._post("/v1/completions", {
        "model": measure.MODEL, "prompt": p + d,
        "max_tokens": 0, "echo": True, "logprobs": 5, "temperature": 0})
    lp = r["choices"][0]["logprobs"]
    mine = lp["token_logprobs"][-1]
    tl = lp["top_logprobs"][-1] or {}
    top = max(tl.items(), key=lambda kv: kv[1]) if tl else (None, 0.0)
    other = max((v for k, v in tl.items() if k != d), default=top[1])
    return mine - other, top[0]


def digit_ok(name, code):
    return [probe(prompt(name, i), d)[1] == d for i, d in enumerate(code)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--joint-iters", type=int, default=250)
    ap.add_argument("--basis", type=int, default=16)
    ap.add_argument("--heads", default="all", choices=["all", "3gram"],
                    help="분해 형식에서는 16헤드를 써도 충돌이 늘지 않는다 (9.3% 동일)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()
    tok = tokenizer()

    # 실행 전 잔여 편집 정리 — 이 세션에서 잔여물이 결론을 세 번 왜곡했다
    cleaned = 0
    for i2, d2 in enumerate(a.code):
        ids2 = encode(prompt(a.name, i2) + d2)
        for k2 in range(2, len(ids2)):
            ad2 = addresses(ids2, k2)
            n2 = live.diff_from_disk(ad2)
            if n2:
                live.restore_from_disk(ad2)
                cleaned += n2
    if cleaned:
        print(f"(실행 전 잔여 편집 {cleaned}행 복원)")

    ok0 = digit_ok(a.name, a.code)
    print(f"=== {a.name} {a.code} ===")
    print("편집 전 " + " ".join(f"{IDX[i]}:{'O' if v else 'X'}"
                              for i, v in enumerate(ok0)) + f"  {sum(ok0)}/6")

    opts = []
    # 1단계: 자리별 독립 window (index 5, ' A' 포함)
    for i, d in enumerate(a.code):
        p = prompt(a.name, i)
        ids = encode(p + d)
        pos = len(encode(p)) - 1
        addrs = addresses(ids, pos)

        def mk(pp, dd):
            def f():
                m, top = probe(pp, dd)
                pen = 0.0 if (top and top.strip().isdigit()) else -1.0
                return min(m, 1.5) + pen
            return f
        HS = ALL_HEADS if a.heads == "all" else TRIGRAM_HEADS
        o = RowOptimizer(addrs, unembed_p(ids[pos + 1]), mk(p, d),
                         n_basis=a.basis, seed=100 + i, heads=HS)
        o.run(n_steps=a.iters, verbose=False)
        opts.append(o)
    ok1 = digit_ok(a.name, a.code)
    print("자리별 후 " + " ".join(f"{IDX[i]}:{'O' if v else 'X'}"
                               for i, v in enumerate(ok1)) + f"  {sum(ok1)}/6")

    # 2단계: 6개 질의가 공유하는 이름 window 를 **동시에** 최적화
    p0 = prompt(a.name, 0)
    ids0 = encode(p0 + a.code[0])
    last = len(encode(p0)) - 1
    shared = [k for k in range(2, last)]        # 이름 안쪽 (인덱스 토큰 제외)
    print(f"공유 window {len(shared)}개: " +
          ", ".join(repr("".join(reversed([tok.decode([ids0[k - j]])
                                           for j in range(3)]))) for k in shared))
    sites = []
    for k in shared:
        ad = addresses(ids0, k)
        sel = np.asarray(ad)[TRIGRAM_HEADS]
        sites.append((ad, read_rows(sel).reshape(-1), TRIGRAM_HEADS))

    def joint_obj():
        s = 0.0
        for i, d in enumerate(a.code):
            m, top = probe(prompt(a.name, i), d)
            s += min(m, 1.0) + (0.0 if (top and top.strip().isdigit()) else -1.0)
        return s

    jo = JointOptimizer(sites, joint_obj, n_basis=8, seed=5)
    if not jo.check_identity():
        sys.exit("theta=0 이 원본을 재현하지 못한다 — 시작점 버그")
    jo.run(n_steps=a.joint_iters, log_every=max(1, a.joint_iters // 3))
    opts.append(jo)

    ok2 = digit_ok(a.name, a.code)
    print("공유 후   " + " ".join(f"{IDX[i]}:{'O' if v else 'X'}"
                                for i, v in enumerate(ok2)) + f"  {sum(ok2)}/6")
    print(f"\n{sum(ok0)}/6 -> {sum(ok1)}/6 -> **{sum(ok2)}/6**   전체 정답 {all(ok2)}")

    if a.out:
        json.dump({"name": a.name, "code": a.code, "before": sum(ok0),
                   "after_perdigit": sum(ok1), "after_shared": sum(ok2),
                   "full": all(ok2)}, open(a.out, "w"), ensure_ascii=False)
    if not a.keep:
        for o in opts:
            o.restore()
        print("복원 완료")


if __name__ == "__main__":
    main()
