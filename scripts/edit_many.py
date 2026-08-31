"""단일 편집을 N회 반복하고, 매번 이전 사실들이 살아있는지 검사한다.

목표 검증의 본체.  "한국 주식 종목명을 모두 틀리지 않고 대답하는가"를
실제로 재려면 (1) 편집이 누적되는지, (2) 나중 편집이 앞 편집을 망가뜨리는지
(순차 편집의 경로 의존성) 둘 다 봐야 한다.

    python scripts/edit_many.py --n 12 --iters 50

주소가 겹치지 않는 종목만 고른다 — 겹치는 종목은 같은 행을 읽으므로
원리적으로 함께 저장될 수 없다 (results/03-addressability.md).
"""
import argparse, collections, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram import live, measure
from engram.core import encode, addresses
from engram.edit import plan, unembed_p, unembed_basis
from engram.opt import RowOptimizer

FMT = "종목코드 {name}"
ROOT = os.path.join(os.path.dirname(__file__), "..")


def addr_key(name, code):
    """첫 코드 토큰을 결정하는 위치(= trigger 마지막 토큰, index p-1)의 주소.

    trigger_uniqueness.py 와 같은 위치를 봐야 상한 계산과 일치한다.
    """
    trig = FMT.format(name=name)
    ids = encode(trig + " " + code)
    p = len(encode(trig))
    return tuple(addresses(ids, p - 1).tolist())


def pick_unique(candidates, n):
    """주소가 다른 종목과 겹치지 않는 것만 고른다."""
    keys = collections.Counter()
    per = {}
    for r in candidates:
        try:
            k = addr_key(r["name"], r["code"])
        except Exception:
            continue
        keys[k] += 1
        per[r["name"]] = k
    out = [r for r in candidates if per.get(r["name"]) and keys[per[r["name"]]] == 1]
    return out[:n]


def edit_one(name, code, steps, iters, basis, heads=None, objective="margin",
             basis_mode="random"):
    trig, ans = FMT.format(name=name), " " + code
    scorer = measure.AnswerScorer(trig, ans)
    base = sum(scorer.vector())
    base_mrg = scorer.margins()
    pl = plan(trig, ans)
    opts = []
    for i in steps:
        if i >= len(pl["steps"]):
            continue
        s = pl["steps"][i]
        obj = (scorer.margin_objective(i) if objective == "margin"
               else scorer.tail(i))
        # 정보 기저(목표·경쟁 토큰의 W_V^-1 u 방향)는 기본으로 끈다.
        # 선행연구 실측상 실제로 작동하는 행은 gold unembedding 방향과
        # cos 0.16 으로 **정렬돼 있지 않다** — 그 부분공간에 가두면 오히려 나빠진다.
        B = None
        if basis_mode == "informed":
            comps = scorer.competitors(i, k=4)
            comp_ids = [c for t in comps for c in encode(t)][:4]
            B = unembed_basis(s["target_id"], comp_ids)
        from engram.opt import heads_for_step
        o = RowOptimizer(s["addrs"], unembed_p(s["target_id"]), obj,
                         n_basis=basis, seed=1000 + i,
                         heads=heads_for_step(i, heads), basis=B)
        o.run(n_steps=iters, verbose=False)
        opts.append(o)
    final = sum(scorer.vector())
    final_mrg = scorer.margins()
    gen = measure.generate(trig, 10)
    return {"name": name, "code": code, "baseline": base, "final": final,
            "baseline_margins": base_mrg, "final_margins": final_mrg,
            "neg_margins": int(sum(1 for m in final_mrg if m <= 0)),
            "gen": gen, "hit": code in gen}, opts


def check(name, code):
    trig = FMT.format(name=name)
    gen = measure.generate(trig, 10)
    return code in gen, gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--steps", default="0,1,2,3,4")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--basis", type=int, default=32)
    ap.add_argument("--only-reachable", action="store_true",
                    help="도달 가능 예측 종목만 편집해 실제 달성치를 측정")
    ap.add_argument("--threshold", type=float, default=-10.0)
    ap.add_argument("--basis-mode", default="random", choices=["random","informed"])
    ap.add_argument("--stratified", action="store_true",
                    help="도달 가능 구간 전체에서 고르게 표집 (선택 편향 제거)")
    ap.add_argument("--heads", default="adaptive", choices=["adaptive","all","3gram","2gram"])
    ap.add_argument("--objective", default="margin", choices=["margin", "tail"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "edit-many.json"))
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",")]

    if not live.available():
        sys.exit("패치 훅 없음. ENGRAM_HOOK=1 로 서버를 띄웠는가?")
    measure.warmup()

    # reachability.json 의 자리별 margin 으로 "고칠 수 있다/없다"를 미리 예측하고,
    # 두 부류를 섞어 뽑아 예측이 맞는지 검증한다 (results/05-channel-capacity.md).
    rpath = os.path.join(ROOT, "results", "reachability.json")
    reach = json.load(open(rpath))["rows"] if os.path.exists(rpath) else []
    by_name = {r["name"]: r for r in reach}
    krx = {r["회사명"]: r["종목코드"] for r in
           json.load(open(os.path.join(ROOT, "data", "krx.json")))}

    def worst(name):
        r = by_name.get(name)
        return min(r["margins"]) if r else None

    cands = [{"name": n, "code": krx[n], "worst": worst(n)}
             for n in by_name if n in krx and worst(n) is not None]
    cands = [c for c in cands if c["worst"] <= 0]          # 이미 맞는 건 제외
    reachable = sorted([c for c in cands if c["worst"] > args.threshold],
                       key=lambda c: -c["worst"])
    unreach = sorted([c for c in cands if c["worst"] < -3.0],
                     key=lambda c: c["worst"])
    if args.only_reachable:
        # 달성 가능 상한이 아니라 **실제 달성치**를 재려면 도달 가능 집합만 편집한다.
        # 기본은 쉬운 순(worst margin 큰 순)이라 결과가 부풀려진다 —
        # --stratified 로 도달 가능 구간 전체에서 고르게 뽑아야 정직한 수치가 나온다.
        pool = pick_unique(reachable, len(reachable))
        if args.stratified and len(pool) > args.n:
            lo, hi = args.threshold, max(c["worst"] for c in pool)
            edges = np.linspace(lo, hi, args.n + 1)
            chosen = []
            for a, b in zip(edges[:-1], edges[1:]):
                band = [c for c in pool if a <= c["worst"] < b and c not in chosen]
                if band:
                    chosen.append(band[len(band) // 2])
            chosen = chosen[:args.n]
        else:
            chosen = pool[:args.n]
    else:
        half = max(1, args.n // 2)
        chosen = pick_unique(reachable, half) + pick_unique(unreach, args.n - half)
    thr = args.threshold
    print(f"도달 가능 예측 {len([c for c in chosen if c['worst'] > thr])}개 + "
          f"불가능 예측 {len([c for c in chosen if c['worst'] <= thr])}개 = {len(chosen)}개\n")
    for c in chosen:
        print(f"  {c['name']:<12s} {c['code']}  최악 margin {c['worst']:+.2f}  "
              f"예측 {'가능' if c['worst'] > -1 else '불가능'}")
    print()

    log, all_opts = [], []
    t0 = time.time()
    for j, r in enumerate(chosen):
        rec, opts = edit_one(r["name"], r["code"], steps, args.iters,
                             args.basis, args.heads, args.objective, args.basis_mode)
        rec["worst_margin_before"] = r["worst"]
        rec["predicted"] = bool(r["worst"] > args.threshold)
        all_opts += opts
        # 이전에 편집한 모든 사실을 다시 검사 (순차 간섭)
        retained = []
        for prev in log:
            ok, gen = check(prev["name"], prev["code"])
            retained.append(ok)
        rec["retained_prev"] = retained
        rec["n_prev_ok"] = int(sum(retained))
        log.append(rec)
        print(f"[{j+1:2d}/{len(chosen)}] {r['name']:<10s} {r['code']}  "
              f"logprob {rec['baseline']:+7.2f} -> {rec['final']:+7.2f}  "
              f"hit={rec['hit']!s:<5s} 음수margin {rec['neg_margins']}  "
              f"이전유지 {rec['n_prev_ok']}/{len(retained)}  "
              f"예측={'가능' if rec['predicted'] else '불가'} "
              f"gen={rec['gen'][:18]!r}", flush=True)
        json.dump({"format": FMT, "steps": steps, "iters": args.iters,
                   "heads": args.heads, "objective": args.objective,
                   "elapsed_s": time.time() - t0, "runs": log},
                  open(args.out, "w"), ensure_ascii=False, indent=2)

    hits = sum(1 for r in log if r["hit"])
    print(f"\n최종: {hits}/{len(log)} 정답 ({100.0*hits/max(1,len(log)):.1f}%)")
    for lab, want in [("도달 가능 예측", True), ("불가능 예측", False)]:
        g = [r for r in log if r["predicted"] == want]
        if g:
            h = sum(1 for r in g if r["hit"])
            print(f"  {lab}: {h}/{len(g)} 정답")
    print(f"기록 -> {args.out}")
    print("편집은 유지된다.  되돌리려면 restore_all.py 를 쓰거나 서버를 재기동하라.")
    np.savez(os.path.join(ROOT, "engram", "backups", "edit-many-originals.npz"),
             addrs=np.concatenate([o.addrs for o in all_opts]),
             rows=np.concatenate([o.original for o in all_opts]))


if __name__ == "__main__":
    main()
