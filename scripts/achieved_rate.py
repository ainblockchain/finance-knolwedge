"""달성률 실측 — 실제로 틀리는 종목을 무작위 표집해 편집 성공률을 잰다.

margin 으로 도달 가능성을 추정하지 않는다.  선택도 판정도 **생성** 으로만 한다.
  1. 무작위 종목의 생성이 틀리는지 확인 (틀리는 것만 대상)
  2. 깨끗한 테이블에서 편집
  3. 생성으로 판정
  4. 복원하고 잔여 확인
"""
import argparse, json, os, re, sys, time
sys.stdout.reconfigure(line_buffering=True)   # 진행을 실시간으로 본다
import concurrent.futures as cf
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram import live, measure
from engram.core import encode, addresses, read_rows
from engram.edit import plan, unembed_p
from engram.opt import RowOptimizer, heads_for_step, run_with_restarts, JointOptimizer
from engram.template import trigger

ROOT = os.path.join(os.path.dirname(__file__), "..")
CODE = re.compile(r"([0-9][0-9A-Z]{5})")


def is_wrong(rec):
    n, c = rec["회사명"], rec["종목코드"]
    try:
        g = measure.generate(trigger(n, "plain"), 8)
        m = CODE.search(g)
        return None if (m and m.group(1) == c) else (n, c, g)
    except Exception:
        return None


def _sites(gs, trig, ans, steps, pre_steps):
    pl = plan(trig, ans, pre_steps=pre_steps)
    by = {st["step"]: st for st in pl["steps"]}
    out = []
    for i in steps:
        if i not in by:
            continue
        s = by[i]
        e0 = (read_rows(s["addrs"]).reshape(-1) if i < 0
              else unembed_p(s["target_id"]))
        out.append((i, s["addrs"], e0, heads_for_step(i, "adaptive")))
    return out


def _run_joint(gs, sites, iters, basis):
    opt = JointOptimizer([(a, e, h) for _, a, e, h in sites],
                         gs.staged_objective(), n_basis=basis, seed=1234)
    opt.run(n_steps=iters, verbose=False)
    return [opt]


def _run_per_step(gs, sites, iters, basis, restarts):
    opts = []
    for i, addrs, e0, heads in sites:
        _, o = run_with_restarts(addrs, e0, gs.staged_objective(),
                                 n_restarts=restarts, n_steps=iters,
                                 n_basis=basis, seed0=1000 + i, heads=heads)
        opts.append(o)
    return opts


def edit(name, code, steps, iters, basis, pre_steps=0, restarts=1,
         strategy="best"):
    """strategy: joint | per_step | best

    어느 방식도 지배적이지 않다 — 조정이 병목이면 Joint 가, 탐색 차원이
    병목이면 단계별이 낫고 사례마다 갈린다.  `best` 는 둘 다 돌려 이긴 쪽만
    남긴다.  진 쪽은 원본으로 되돌리므로 각각보다 엄격히 낫다.
    """
    trig, ans = trigger(name, "plain"), " " + code
    gs = measure.GenScorer(trig, ans)
    before = gs.matched()
    sites = _sites(gs, trig, ans, steps, pre_steps)

    # 승자를 다시 돌리면 비용이 두 배가 된다.  Joint 가 싸므로(전체 iters 1회분)
    # 먼저 돌려보고, 정답이면 거기서 끝낸다.  아니면 되돌리고 단계별로 간다.
    tried = {}
    if strategy in ("joint", "best"):
        opts = _run_joint(gs, sites, iters, basis)
        tried["joint"] = {"matched": gs.matched(), "hit": gs.hit()}
        if strategy == "joint" or tried["joint"]["hit"]:
            return {"name": name, "code": code, "matched_before": before,
                    "matched_after": gs.matched(), "n_ans": gs.n_ans,
                    "hit": gs.hit(), "winner": "joint", "tried": tried}, opts
        joint_matched = tried["joint"]["matched"]
        for o in opts:
            o.restore()
    else:
        joint_matched = -1

    opts = _run_per_step(gs, sites, iters, basis, restarts)
    tried["per_step"] = {"matched": gs.matched(), "hit": gs.hit()}
    winner = "per_step"
    # 단계별이 Joint 보다 못하면 되돌리고 Joint 를 다시 적용한다 (이때만 재실행)
    if strategy == "best" and not gs.hit() and gs.matched() < joint_matched:
        for o in opts:
            o.restore()
        opts = _run_joint(gs, sites, iters, basis)
        winner = "joint"

    return {"name": name, "code": code, "matched_before": before,
            "matched_after": gs.matched(), "n_ans": gs.n_ans,
            "hit": gs.hit(), "winner": winner, "tried": tried}, opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--names", default="", help="쉼표로 특정 종목만 (name:code)")
    ap.add_argument("--pool", type=int, default=120)
    ap.add_argument("--steps", default="-4,-3,-2,-1,0,1,2,3,4,5,6")
    ap.add_argument("--pre-steps", type=int, default=4)
    ap.add_argument("--strategy", default="best",
                    choices=["best", "joint", "per_step"],
                    help="best = 둘 다 돌려 이긴 쪽만 남긴다")
    ap.add_argument("--restarts", type=int, default=1,
                    help="단계별 재시작 횟수. 분산이 커서 짧게 여러 번이 낫다")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--basis", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "achieved-rate.json"))
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",")]
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()

    if args.names:
        targets = [(x.split(":")[0], x.split(":")[1], "") for x in args.names.split(",")]
        print(f"지정 종목 {len(targets)}개\n", flush=True)
        log, t0 = [], time.time()
        for j, (n, c, _) in enumerate(targets):
            rec, opts = edit(n, c, steps, args.iters, args.basis, args.pre_steps, args.restarts,
                         strategy=args.strategy)
            for o in opts:
                o.restore()
            log.append(rec)
            tried = " ".join(f"{k}:{v['matched']}{'*' if v['hit'] else ''}"
                             for k, v in rec.get("tried", {}).items())
            print(f"[{j+1:2d}/{len(targets)}] {n:<16s} {c}  일치 "
                  f"{rec['matched_before']}/{rec['n_ans']} -> {rec['matched_after']}/{rec['n_ans']}  "
                  f"hit={rec['hit']}  [{tried}]", flush=True)
            json.dump({"iters": args.iters, "steps": steps, "runs": log},
                      open(args.out, "w"), ensure_ascii=False, indent=2)
        h = sum(1 for r in log if r["hit"])
        print(f"\n달성률: {h}/{len(log)}")
        return

    recs = json.load(open(os.path.join(ROOT, "data", "krx.json")))
    rng = np.random.default_rng(7)
    pool = [recs[i] for i in rng.choice(len(recs), args.pool, replace=False)]
    with cf.ThreadPoolExecutor(16) as ex:
        wrong = [w for w in ex.map(is_wrong, pool) if w]
    print(f"무작위 {len(pool)}개 중 생성이 틀리는 종목 {len(wrong)}개 "
          f"({100.0*len(wrong)/len(pool):.0f}%)\n", flush=True)
    targets = wrong[:args.n]

    log, t0 = [], time.time()
    for j, (n, c, g) in enumerate(targets):
        rec, opts = edit(n, c, steps, args.iters, args.basis, args.pre_steps, args.restarts,
                         strategy=args.strategy)
        for o in opts:
            o.restore()
        log.append(rec)
        print(f"[{j+1:2d}/{len(targets)}] {n:<14s} {c}  일치 "
              f"{rec['matched_before']}/{rec['n_ans']} -> {rec['matched_after']}/{rec['n_ans']}  "
              f"hit={rec['hit']}", flush=True)
        json.dump({"iters": args.iters, "steps": steps,
                   "elapsed_s": time.time() - t0, "runs": log},
                  open(args.out, "w"), ensure_ascii=False, indent=2)

    h = sum(1 for r in log if r["hit"])
    print(f"\n실제로 틀리는 종목에서의 달성률: {h}/{len(log)} = "
          f"{100.0*h/max(1,len(log)):.0f}%")


if __name__ == "__main__":
    main()
