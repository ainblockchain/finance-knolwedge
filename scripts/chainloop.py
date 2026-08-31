"""체인 반복 통과 — 앞을 고치면 뒤 문맥이 바뀌므로 수렴할 때까지 돈다.

한 번 훑어 고치면 앞선 편집이 뒤 자리의 문맥을 바꿔 멀쩡하던 자리가 깨진다
(실측: 3O 7X 8X 3O 4O 0O -> 3O 7O 8O 3X 4X 0O).
그래서 매 통과마다 전체 체인을 다시 점검하고 틀린 자리만 고친다.

자리당 채널은 넉넉하다 — 실측 -7.06 -> +5.75 (12.8 nat).
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
from engram.core import encode, addresses
from engram.edit import unembed_p
from engram.opt import RowOptimizer, ALL_HEADS

ROOT = os.path.join(os.path.dirname(__file__), "..")
FMT = "종목코드 {name} "


def margin(p, t):
    r = measure._post("/v1/completions", {
        "model": measure.MODEL, "prompt": p,
        "max_tokens": 1, "temperature": 0, "logprobs": 10})
    tl = r["choices"][0]["logprobs"]["top_logprobs"][0] or {}
    mine = tl.get(t)
    if mine is None:
        e = measure._post("/v1/completions", {
            "model": measure.MODEL, "prompt": p + t,
            "max_tokens": 0, "echo": True, "logprobs": 5, "temperature": 0})
        mine = e["choices"][0]["logprobs"]["token_logprobs"][-1]
    other = max((v for k, v in tl.items() if k != t), default=-20.0)
    top = max(tl.items(), key=lambda kv: kv[1])[0] if tl else None
    return mine - other, top


def run(name, code, passes, iters, basis, verbose=True):
    base = FMT.format(name=name)
    opts = []

    # 공백을 떼면 모델이 숫자가 아니라 산문을 낸다 (', 2차전지 장비 전문').
    # teacher forcing 체인과 같은 프롬프트를 써야 두 측정이 일치한다.
    gs = measure.GenScorer(base, code)

    def chain():
        """생성 경로에서 잰다.

        teacher forcing 으로 6/6 이어도 자유 생성은 다를 수 있다 — 생성은 자기가
        만든 접두를 따라가므로 한 자리가 갈리면 뒤 문맥 전체가 달라진다.
        실측: 체인 6/6 인데 생성은 '3783860'.  그래서 생성 margin 으로 재고,
        고칠 자리도 **생성이 갈리는 자리**로 잡는다.
        """
        m = gs.margins()
        return [(m[k], code[k] if m[k] > 0 else None) for k in range(len(code))]

    st = chain()
    if verbose:
        print("시작 " + " ".join(f"{code[k]}{'O' if c[1]==code[k] else 'X'}"
                                for k, c in enumerate(st)))
    for it in range(passes):
        st = chain()
        bad = [k for k, c in enumerate(st) if c[1] != code[k]]
        if not bad:
            break
        k = bad[0]                      # 가장 앞의 틀린 자리부터
        p, t = base + code[:k], code[k]
        ids = encode(p + t)
        pos = len(encode(p)) - 1
        a = addresses(ids, pos)

        def obj():
            """**생성 경로에서 재되, 그 자리 margin 단독**을 목적함수로.

            staged(일치 접두 + 첫 실패 margin) 는 접두 길이가 정수라 계단이고,
            남는 margin 항은 0.0625 양자화에 걸려 신호가 거의 없다 —
            같은 자리를 12회 시도해도 안 움직였다.
            teacher forcing 에서 단일 자리 목적함수는 -4.06 -> +0.50 으로 통했다.
            """
            m = gs.margins()
            v = min(m[k], 2.0)
            # 앞 자리를 깨뜨리면 벌점 (뒤 자리는 문맥이 바뀌므로 세지 않는다)
            for j in range(k):
                if m[j] <= 0:
                    v -= 1.0
            return v

        o = RowOptimizer(a, unembed_p(ids[pos + 1]), obj,
                         n_basis=basis, seed=5 + it, heads=ALL_HEADS)
        o.run(n_steps=iters, verbose=False)
        opts.append(o)
        if verbose:
            st2 = chain()
            print(f"  통과{it+1} 자리{k} 고침 -> " +
                  " ".join(f"{code[j]}{'O' if c[1]==code[j] else 'X'}"
                           for j, c in enumerate(st2)))
    st = chain()
    ok = all(c[1] == code[k] for k, c in enumerate(st))
    return ok, st, opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--passes", type=int, default=12)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--basis", type=int, default=32)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()
    t0 = time.time()
    ok, st, opts = run(a.name, a.code, a.passes, a.iters, a.basis)
    gen = measure.generate(FMT.format(name=a.name), 8)
    print(f"\n최종 {'성공' if ok else '실패'}  생성 {gen!r}  ({time.time()-t0:.0f}초, "
          f"{len(opts)}회 편집)")
    if a.out:
        json.dump({"name": a.name, "code": a.code, "ok": bool(ok),
                   "gen": gen, "edits": len(opts)}, open(a.out, "w"),
                  ensure_ascii=False)
    if not a.keep:
        for o in opts:
            o.restore()
        print("복원 완료")


if __name__ == "__main__":
    main()
