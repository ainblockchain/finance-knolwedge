"""전 지점 동시 최적화(Joint OPT)로 사실 하나를 편집한다.

단계별 독립 최적화는 뒤를 고치며 앞을 깨뜨린다.  모든 지점의 theta 를 하나로
이어붙여 SPSA 를 돌리면 평가 1회가 전체 구성을 재므로 그 문제가 없다.

    python scripts/edit_joint.py --name HLB생명과학 --code 067630 \
        --fmt "{name}의 종목코드는" --pre-steps 5 "--steps=-5,-4,-3,-2"
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram import live, measure
from engram.core import read_rows
from engram.edit import plan, unembed_p
from engram.opt import JointOptimizer, heads_for_step
from engram.template import trigger as make_trigger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--style", default=None, choices=["plain", "head1", "hash2"])
    ap.add_argument("--fmt", default="종목코드 {name}")
    ap.add_argument("--steps", default="0,1,2,3,4")
    ap.add_argument("--pre-steps", type=int, default=0)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--basis", type=int, default=24)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trig = (make_trigger(args.name, args.style) if args.style
            else args.fmt.format(name=args.name))
    ans = " " + args.code
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()

    gs = measure.GenScorer(trig, ans)
    print(f"=== {trig!r} -> {ans!r} ===")
    print(f"편집 전  생성 {measure.generate(trig, 10)!r}")
    print("         margin " + " ".join(f"{v:+.2f}" for v in gs.margins()))
    print(f"         일치 접두 {gs.matched()}/{gs.n_ans}\n")

    pl = plan(trig, ans, pre_steps=args.pre_steps)
    by = {s["step"]: s for s in pl["steps"]}
    want = [int(x) for x in args.steps.split(",")]
    sites = []
    for i in want:
        if i not in by:
            continue
        s = by[i]
        e0 = (read_rows(s["addrs"]).reshape(-1) if i < 0
              else unembed_p(s["target_id"]))
        sites.append((s["addrs"], e0, heads_for_step(i, "adaptive")))
    print(f"편집 지점 {len(sites)}개: {[i for i in want if i in by]}\n")

    opt = JointOptimizer(sites, gs.staged_objective(),
                         n_basis=args.basis, seed=1234)
    res = opt.run(n_steps=args.iters)

    gen = measure.generate(trig, 10)
    hit = args.code in gen
    print(f"\n편집 후  생성 {gen!r}   정답포함={hit}")
    print("         margin " + " ".join(f"{v:+.2f}" for v in gs.margins()))
    print(f"         일치 접두 {gs.matched()}/{gs.n_ans}")

    if args.out:
        json.dump({"trigger": trig, "answer": ans, "sites": len(sites),
                   "hit": hit, "gen": gen, **res},
                  open(args.out, "w"), ensure_ascii=False, indent=2)
    if not args.keep:
        print(f"복원 {opt.restore()}행")


if __name__ == "__main__":
    main()
