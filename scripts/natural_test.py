"""템플릿으로 편집한 뒤, 행을 복원하지 않은 채 자연스러운 질문에도 답이 나오는지.

편집은 "종목코드 {name} " 템플릿에서만 이뤄진다.  질문은 토큰 열이 다르므로
같은 행을 지나가지 않을 수 있다 — 그러면 편집 효과가 전혀 안 실린다.
이 스크립트는 그 전이(transfer)를 직접 잰다.
"""
import argparse, os, re, sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram import live, measure
from engram.core import encode, addresses
from engram.edit import unembed_p
from engram.opt import RowOptimizer, ALL_HEADS

FMT = "종목코드 {name} "
CODE = re.compile(r"([0-9][0-9A-Z]{5})")

QUESTIONS = [
    "{name} 종목코드는 ",
    "{name}의 종목코드는 ",
    "{name} 종목코드가 뭐야? ",
    "{name}의 종목코드를 알려줘. ",
    "Q: {name} 종목코드는?\nA: ",
    "{name} 코드는 ",
    "{name} ({code_label}",
]

def ask(name):
    out = []
    for q in QUESTIONS:
        p = q.format(name=name, code_label="")
        g = measure.generate(p, 10)
        m = CODE.search(g)
        out.append((p.replace("\n", "\\n"), g.replace("\n", "\\n"), m.group(1) if m else None))
    return out

def show(rows, code):
    hit = 0
    for p, g, c in rows:
        ok = c == code
        hit += ok
        print(f"  {'O' if ok else 'X'}  {p!r:36s} -> {g!r}")
    print(f"  정답 {hit}/{len(rows)}")
    return hit

def solve(name, code, passes, iters, basis):
    base = FMT.format(name=name)
    gs = measure.GenScorer(base, code)
    opts = []
    for it in range(passes):
        m = gs.margins()
        bad = [k for k in range(len(code)) if m[k] <= 0]
        if not bad:
            break
        k = bad[0]
        p, t = base + code[:k], code[k]
        ids = encode(p + t)
        pos = len(ids) - 2
        a = addresses(ids, pos)
        def obj():
            mm = gs.margins()
            v = min(mm[k], 2.0)
            for j in range(k):
                if mm[j] <= 0: v -= 1.0
            return v
        o = RowOptimizer(a, unembed_p(ids[pos + 1]), obj, n_basis=basis, heads=ALL_HEADS)
        o.run(n_steps=iters)
        opts.append(o)
        print(f"  통과{it+1} 자리{k} -> 일치 {gs.matched()}/{len(code)}")
    return gs.matched() == len(code), opts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--code", required=True)
    ap.add_argument("--passes", type=int, default=8); ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--basis", type=int, default=32)
    a = ap.parse_args()
    measure.warmup()
    print(f"== 편집 전: 템플릿 == {measure.generate(FMT.format(name=a.name), 8)!r}")
    print("== 편집 전: 자연 질문 ==")
    before = show(ask(a.name), a.code)
    print("== 템플릿 편집 ==")
    ok, opts = solve(a.name, a.code, a.passes, a.iters, a.basis)
    print(f"템플릿 {'성공' if ok else '실패'}: {measure.generate(FMT.format(name=a.name), 8)!r}")
    print("== 편집 후: 자연 질문 (행 유지) ==")
    after = show(ask(a.name), a.code)
    print(f"\n전이: 자연 질문 정답 {before} -> {after} / {len(QUESTIONS)}")
    for o in opts: o.restore()
    print("복원 완료")

if __name__ == "__main__":
    main()
