"""단일 사실 하나를 Engram 테이블에 pin-point 편집하고 검증한다.

    python scripts/edit_one_fact.py --name 닷밀 --code 464580 --steps 1,2 --iters 200

절차
  1. 편집 전 기준선 측정 (greedy 생성 + answer logprob)
  2. plan() 으로 편집 단계와 16행 주소 산출
  3. 지정한 단계마다 SPSA 로 행 값 최적화 (forward-only)
  4. 편집 후 재측정
  5. 지역성 검사 — 건드리지 않은 프롬프트들이 그대로인지
  6. --keep 없으면 원본 행 복원
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram import live, measure
from engram.core import consts, read_rows, encode
from engram.edit import plan, unembed_p, unembed_basis
from engram.opt import RowOptimizer

LOCALITY_PROBES = [
    ("대한민국의 수도는", " 서울"),
    ("삼성전자 종목코드는", " 005930"),
    ("SK하이닉스 종목코드는", " 000660"),
    ("The capital of France is", " Paris"),
    ("2 + 2 =", " 4"),
]


_HEAT_BASE = None


def _heat_baseline(n=200, seed=7):
    """헤드별 (평균, 표준편차) — 무작위 행 노름.  한 번만 계산해 재사용한다."""
    global _HEAT_BASE
    if _HEAT_BASE is None:
        C = consts()
        rng = np.random.default_rng(seed)
        _HEAT_BASE = []
        for k in range(C.n_heads_total):
            lo = int(C.head_offset[k]); hi = lo + int(C.head_vocab[k])
            b = np.linalg.norm(read_rows(rng.integers(lo, hi, n)), axis=1)
            _HEAT_BASE.append((b.mean(), b.std()))
    return _HEAT_BASE


def heat(step):
    """행 열기 z-점수 (같은 헤드 범위 무작위 행 대비).  높을수록 자주 밟힌 행."""
    base = _heat_baseline()
    n = np.linalg.norm(read_rows(step["addrs"]), axis=1)
    z = [(n[k] - m) / sd for k, (m, sd) in enumerate(base)]
    return float(np.mean(z[:8])), float(np.mean(z[8:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--steps", default="", help="편집할 step 번호 (쉼표). 음수 = trigger 내부")
    ap.add_argument("--pre-steps", type=int, default=0,
                    help="trigger 내부 편집 지점 수. 어텐션을 통해 답 위치에 작용하는 추가 채널")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--basis", type=int, default=48)
    ap.add_argument("--keep", action="store_true", help="복원하지 않고 남긴다")
    ap.add_argument("--heads", default="adaptive", choices=["adaptive","all","3gram","2gram"],
                    help="편집할 헤드. 3gram(8~15)만 쓰면 부수 피해가 급감한다")
    ap.add_argument("--objective", default="staged", choices=["staged","genmargin","margin","tail"],
                    help="margin: 목표토큰-2등 (greedy 정답을 직접 겨냥). tail: logprob 합")
    ap.add_argument("--basis-mode", default="random", choices=["random","informed"],
                    help="random 권장 — informed 는 unembedding 부분공간에 갇혀 성능이 나쁘다")
    ap.add_argument("--style", default=None, choices=["plain","head1","hash2"],
                    help="트리거 템플릿 (engram/template.py). --fmt 보다 우선")
    ap.add_argument("--fmt", default="종목코드 {name}",
                    help="trigger 형식. 기본은 이름이 코드 바로 앞 (주소 구분 가능)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.style:
        from engram.template import trigger as make_trigger
        trigger = make_trigger(args.name, args.style)
    else:
        trigger = args.fmt.format(name=args.name)
    answer = f" {args.code}"

    if not live.available():
        sys.exit("패치 훅에 연결할 수 없다. ENGRAM_HOOK=1 로 서버를 띄웠는가?")

    print(f"=== {trigger!r} -> {answer!r} ===\n")

    measure.warmup()
    scorer = measure.AnswerScorer(trigger, answer)
    # 생성 경로에서 직접 재는 채점기 — teacher forcing 과 결과가 다르다
    gscorer = measure.GenScorer(trigger, answer)
    base_vec = scorer.vector()
    base_lp = sum(base_vec)
    base_mrg = scorer.margins()
    base_gm = gscorer.margins()
    base_gen = measure.generate(trigger, 12)
    print(f"편집 전:  logprob {base_lp:+.4f}   생성 {base_gen!r}")
    print("          토큰별 logprob " + " ".join(f"{v:+.2f}" for v in base_vec))
    print("          토큰별 margin  " + " ".join(f"{v:+.2f}" for v in base_mrg))
    print("          생성 margin  " + " ".join(f"{v:+.2f}" for v in base_gm))
    print(f"          생성 일치 접두 {gscorer.matched()}/{gscorer.n_ans}\n")

    pl = plan(trigger, answer, pre_steps=args.pre_steps)
    print("step  target  문맥                         2gram_z  3gram_z")
    heats = []
    for s in pl["steps"]:
        z2, z3 = heat(s)
        heats.append((z2, z3))
        print(f"  {s['step']:>2}  {s['target']!r:8s} {''.join(reversed(s['ctx']))!r:26s} "
              f"{z2:+7.2f}  {z3:+7.2f}")

    by_step = {st["step"]: st for st in pl["steps"]}
    if args.steps:
        chosen = [int(x) for x in args.steps.split(",")]
    else:  # 문맥에 숫자 아닌 토큰이 하나라도 있는 단계만
        chosen = [s["step"] for s in pl["steps"]
                  if any(not c.strip().isdigit() and c.strip() for c in s["ctx"])]
    print(f"\n편집할 단계: {chosen}\n")

    # 프롬프트마다 측정 노이즈가 다르다 (대부분 0, 일부 ~0.1).  먼저 각 프롬프트의
    # 노이즈 바닥을 재고, 그보다 큰 변화만 실제 부수 피해로 센다.
    probe_scorers = {t: measure.AnswerScorer(t, a) for t, a in LOCALITY_PROBES}
    probes_before, probe_noise = {}, {}
    for t, sc in probe_scorers.items():
        vs = [sum(sc.vector()) for _ in range(3)]
        probes_before[t] = float(np.median(vs))
        probe_noise[t] = float(max(vs) - min(vs))

    opts, results = [], []
    t0 = time.time()
    for i in chosen:
        if i not in by_step:
            continue
        s = by_step[i]
        print(f"[step {i}] {''.join(reversed(s['ctx']))!r} -> {s['target']!r}")
        # 목적함수: 이 단계 토큰과 그 이후 전부 (conv 결합 때문에 tail 이 맞다)
        if args.objective == "staged":
            obj = gscorer.staged_objective()
        elif args.objective == "genmargin":
            obj = gscorer.objective(i)
        elif args.objective == "margin":
            obj = scorer.margin_objective(i)
        else:
            obj = scorer.tail(i)
        from engram.opt import heads_for_step
        heads = heads_for_step(i, args.heads)
        # 음수 step(trigger 내부)은 목표 토큰이 의미 없으므로 원본 행에서 출발한다.
        e0 = (read_rows(s["addrs"]).reshape(-1) if i < 0
              else unembed_p(s["target_id"]))
        # 정보 기저(목표·경쟁 토큰의 W_V^-1 u 방향)는 기본으로 끈다.
        # 선행연구 실측상 실제로 작동하는 행은 gold unembedding 방향과
        # cos 0.16 으로 **정렬돼 있지 않다** — 그 부분공간에 가두면 오히려 나빠진다.
        B = None
        if args.basis_mode == "informed":
            comps = scorer.competitors(i, k=4)
            comp_ids = [c for t in comps for c in encode(t)][:4]
            B = unembed_basis(s["target_id"], comp_ids)
        o = RowOptimizer(s["addrs"], e0, obj,
                         n_basis=args.basis, seed=1000 + i, heads=heads, basis=B)
        results.append(o.run(n_steps=args.iters, log_every=max(1, args.iters // 5)))
        opts.append(o)
        print()

    final_vec = scorer.vector()
    final_lp = sum(final_vec)
    final_mrg = scorer.margins()
    final_gen = measure.generate(trigger, 12)
    hit = args.code in final_gen
    print(f"편집 후:  logprob {final_lp:+.4f}  (Δ {final_lp-base_lp:+.4f})")
    print("          토큰별 logprob " + " ".join(f"{v:+.2f}" for v in final_vec))
    print("          토큰별 margin  " + " ".join(f"{v:+.2f}" for v in final_mrg))
    print(f"          margin 음수 {sum(1 for m in final_mrg if m<=0)}/{len(final_mrg)} 자리")
    gm = gscorer.margins()
    print("          생성 margin  " + " ".join(f"{v:+.2f}" for v in gm))
    print(f"          생성 일치 접두 {gscorer.matched()}/{gscorer.n_ans}")
    print(f"          생성 {final_gen!r}   정답포함={hit}\n")

    print("지역성 검사 (건드리지 않은 프롬프트):")
    loc_ok, exact, probes_after = True, 0, {}
    for t, a in LOCALITY_PROBES:
        sc = probe_scorers[t]
        after = float(np.median([sum(sc.vector()) for _ in range(3)]))
        probes_after[t] = after
        d = after - probes_before[t]
        tol = max(probe_noise[t], 1e-9)
        beyond = abs(d) > tol
        if abs(d) < 1e-9: exact += 1
        if beyond: loc_ok = False
        mark = "동일" if abs(d) < 1e-9 else ("노이즈내" if not beyond else "**변함**")
        print(f"  {t!r:28s} Δ {d:+.6f}  (노이즈 {probe_noise[t]:.6f})  {mark}")
    print(f"  -> 비트 단위 동일 {exact}/{len(LOCALITY_PROBES)}, "
          f"노이즈 초과 변화 {'없음' if loc_ok else '있음'}\n")

    rec = {
        "trigger": trigger, "answer": answer, "name": args.name, "code": args.code,
        "baseline_logprob": base_lp, "final_logprob": final_lp,
        "baseline_gen": base_gen, "final_gen": final_gen, "hit": hit,
        "steps_edited": chosen, "iters": args.iters, "basis": args.basis,
        "heat": heats, "locality_ok": loc_ok,
        "baseline_vec": base_vec, "final_vec": final_vec,
        "baseline_margins": base_mrg, "final_margins": final_mrg,
        "baseline_gen_margins": base_gm, "final_gen_margins": gm,
        "objective": args.objective, "heads": args.heads,
        "probes_delta": {t: probes_after[t] - probes_before[t] for t, _ in LOCALITY_PROBES},
        "elapsed_s": time.time() - t0,
        "per_step": [{"baseline": r["baseline"], "final": r["final"]} for r in results],
    }
    if args.out:
        json.dump(rec, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"기록 -> {args.out}")

    if not args.keep:
        for o in opts:
            o.restore()
        print("원본 행 복원 완료")
    else:
        print("편집 유지 (--keep)")
    return rec


if __name__ == "__main__":
    main()
