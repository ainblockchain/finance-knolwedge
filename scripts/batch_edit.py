"""병렬 배치 편집 — 처리량을 위한 러너.

지금까지는 종목당 9분씩 **순차로** 돌렸다.  서버는 동시 64 시퀀스를 처리하는데
1개만 쓴 셈이다.  전수 2,804개가 순차로 17.5일, 32병렬이면 13시간이다.

동시성 안전:
  두 종목의 편집 행이 겹치면 서로의 최적화를 망친다.  숫자 문맥 자리
  (step 2 이후)는 수백 개 종목이 공유하므로, **실제 주소를 계산해 서로 겹치지
  않는 종목들만** 한 배치로 묶는다 (greedy).  배치 안에서는 완전 병렬이다.
"""
import argparse, json, os, re, sys, time, threading
import concurrent.futures as cf
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram import live, measure
from engram.core import encode, addresses, read_rows
from engram.edit import plan, unembed_p
from engram.opt import RowOptimizer, heads_for_step
from engram.template import trigger

ROOT = os.path.join(os.path.dirname(__file__), "..")
CODE = re.compile(r"([0-9][0-9A-Z]{5})")
_lock = threading.Lock()


def sites_for(name, code, steps, pre_steps):
    trig, ans = trigger(name, "plain"), " " + code
    pl = plan(trig, ans, pre_steps=pre_steps)
    by = {s["step"]: s for s in pl["steps"]}
    out = []
    for i in steps:
        if i not in by:
            continue
        s = by[i]
        out.append((i, np.asarray(s["addrs"], dtype=np.int64),
                    s.get("target_id")))
    return out


def disjoint_batches(cands, steps, pre_steps, size):
    """행이 겹치지 않는 종목들로 배치를 만든다."""
    prepared = []
    for n, c in cands:
        try:
            st = sites_for(n, c, steps, pre_steps)
        except Exception:
            continue
        used = set()
        for i, a, _ in st:
            used.update(int(x) for x in a[heads_for_step(i, "adaptive")])
        prepared.append((n, c, used))
    batches, cur, cur_used = [], [], set()
    for n, c, used in prepared:
        if len(cur) >= size or (used & cur_used):
            if cur:
                batches.append(cur)
            cur, cur_used = [], set()
        cur.append((n, c))
        cur_used |= used
    if cur:
        batches.append(cur)
    return batches


def is_wrong(rec):
    n, c = rec["회사명"], rec["종목코드"]
    try:
        g = measure.generate(trigger(n, "plain"), 8)
        m = CODE.search(g)
        return None if (m and m.group(1) == c) else (n, c)
    except Exception:
        return None


def edit_one(name, code, steps, pre_steps, iters, basis):
    trig, ans = trigger(name, "plain"), " " + code
    gs = measure.GenScorer(trig, ans)
    before = gs.matched()
    st = sites_for(name, code, steps, pre_steps)
    opts = []
    for i, addrs, tid in st:
        heads = heads_for_step(i, "adaptive")
        e0 = (read_rows(addrs).reshape(-1) if i < 0 else unembed_p(tid))
        o = RowOptimizer(addrs, e0, gs.staged_objective(),
                         n_basis=basis, seed=1000 + i, heads=heads)
        o.run(n_steps=iters, verbose=False)
        opts.append(o)
    rec = {"name": name, "code": code, "matched_before": before,
           "matched_after": gs.matched(), "n_ans": gs.n_ans, "hit": gs.hit()}
    if not rec["hit"]:
        for o in opts:
            o.restore()          # 실패는 되돌려 다음 배치를 깨끗하게 둔다
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=400, help="검사할 종목 수")
    ap.add_argument("--max", type=int, default=200, help="편집 시도 상한")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", default="0,1,2,3,4,5,6")
    ap.add_argument("--pre-steps", type=int, default=0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--basis", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "batch-edit.json"))
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",")]
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()

    recs = json.load(open(os.path.join(ROOT, "data", "krx.json")))
    rng = np.random.default_rng(2026)
    pool = [recs[i] for i in rng.choice(len(recs), min(args.pool, len(recs)), replace=False)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        wrong = [w for w in ex.map(is_wrong, pool) if w]
    print(f"검사 {len(pool)}개 -> 오답 {len(wrong)}개 ({100.0*len(wrong)/len(pool):.0f}%), "
          f"{time.time()-t0:.0f}초")

    cands = wrong[:args.max]
    batches = disjoint_batches(cands, steps, args.pre_steps, args.batch)
    print(f"주소 비충돌 배치 {len(batches)}개 "
          f"(평균 {np.mean([len(b) for b in batches]):.1f}종목/배치)\n")

    log, t0 = [], time.time()
    for bi, batch in enumerate(batches):
        with cf.ThreadPoolExecutor(min(args.workers, len(batch))) as ex:
            futs = {ex.submit(edit_one, n, c, steps, args.pre_steps,
                              args.iters, args.basis): n for n, c in batch}
            for f in cf.as_completed(futs):
                try:
                    log.append(f.result())
                except Exception as e:
                    log.append({"name": futs[f], "error": str(e), "hit": False})
        h = sum(1 for r in log if r.get("hit"))
        el = time.time() - t0
        print(f"배치 {bi+1}/{len(batches)}  누적 {len(log)}종목  정답 {h} "
              f"({100.0*h/len(log):.0f}%)  경과 {el/60:.0f}분  "
              f"속도 {len(log)/(el/3600):.0f}종목/시간")
        json.dump({"iters": args.iters, "steps": steps, "workers": args.workers,
                   "elapsed_s": el, "runs": log}, open(args.out, "w"),
                  ensure_ascii=False, indent=2)

    h = sum(1 for r in log if r.get("hit"))
    print(f"\n최종: {h}/{len(log)} = {100.0*h/max(1,len(log)):.1f}%")


if __name__ == "__main__":
    main()
