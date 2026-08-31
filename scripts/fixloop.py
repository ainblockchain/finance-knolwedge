"""갈림 지점 추적 편집 — 실패하는 자리만 골라 고치고 다시 본다.

66차원 joint SPSA 는 2점 추정 노이즈에 묻혀 초기값을 못 이겼다.
대신 **지금 생성이 갈리는 자리 하나**만 잡아 그 window 를 고친다.
차원이 1/6 로 줄고(한 window ~17차원) 목표가 명확하다.

    while 갈림이 있고 예산이 남아있으면:
        p = 첫 실패 자리
        p 의 window 를 최적화 (목적: 전체 일치 접두 + p 의 margin)
        재평가 -> 갈림이 뒤로 밀리면 계속

앞 자리를 깨뜨리면 목적함수가 떨어지므로 그 방향으로 가지 않는다.
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
from engram.core import read_rows
from engram.edit import plan, unembed_p
from engram.opt import RowOptimizer, TRIGRAM_HEADS, ALL_HEADS
from engram.template import trigger

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--pre-steps", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=14)
    ap.add_argument("--iters", type=int, default=45)
    ap.add_argument("--basis", type=int, default=16)
    ap.add_argument("--heads", default="3gram", choices=["3gram", "all"])
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()
    heads = TRIGRAM_HEADS if a.heads == "3gram" else ALL_HEADS

    trig, ans = trigger(a.name, "plain"), " " + a.code
    gs = measure.GenScorer(trig, ans)
    pl = plan(trig, ans, pre_steps=a.pre_steps)
    by = {s["step"]: s for s in pl["steps"]}

    print(f"=== {a.name} {a.code} ===")
    print(f"편집 전 {measure.generate(trig,10)!r}  일치 {gs.matched()}/{gs.n_ans}")

    opts, t0 = [], time.time()
    stuck = 0
    for r in range(a.rounds):
        m = gs.matched()
        if m >= gs.n_ans:
            break
        # 갈리는 자리 = step m.  그 window 가 없으면(음수쪽) 가장 앞 window 부터
        step = m if m in by else min(by)
        s = by[step]
        e0 = (read_rows(s["addrs"]).reshape(-1) if step < 0
              else unembed_p(s["target_id"]))
        o = RowOptimizer(s["addrs"], e0, gs.staged_objective(),
                         n_basis=a.basis, seed=31 * r + step, heads=heads)
        o.run(n_steps=a.iters, verbose=False)
        opts.append(o)
        m2 = gs.matched()
        print(f"  R{r+1:2d} step {step:>3} {''.join(reversed(s['ctx']))!r:14s} "
              f"일치 {m}->{m2}", flush=True)
        if m2 <= m:
            stuck += 1
            if stuck >= 3:
                print("  (3회 연속 진전 없음 — 중단)")
                break
        else:
            stuck = 0

    gen = measure.generate(trig, 10)
    hit = a.code in gen
    print(f"\n편집 후 {gen!r}  일치 {gs.matched()}/{gs.n_ans}  정답={hit}  "
          f"({time.time()-t0:.0f}초, window {len(opts)}회 편집)")
    if a.out:
        json.dump({"name": a.name, "code": a.code, "hit": hit, "gen": gen,
                   "matched": gs.matched(), "rounds": len(opts)},
                  open(a.out, "w"), ensure_ascii=False, indent=2)
    if not a.keep:
        for o in opts:
            o.restore()
        print("복원 완료")


if __name__ == "__main__":
    main()
