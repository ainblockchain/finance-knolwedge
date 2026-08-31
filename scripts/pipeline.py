"""대량 편집 파이프라인 — 싼 방법부터 시도하고 실패하면 무거운 방법으로.

관측:
  fixloop (갈림 자리만)      레이언스 42초, BYC 75초 성공 / 다른 건 3~4분 쓰고 실패
  전 window 순차            에프엔에스테크는 이쪽만 성공 (fixloop 로는 2/7)
어느 하나가 지배적이지 않으므로 **싼 것부터** 걸고 실패하면 넘어간다.

병렬:
  성공한 편집은 행 8~40개뿐이라 종목 간 충돌이 드물다.  그래도 배치마다
  실제 주소를 계산해 겹치지 않는 종목만 묶는다.
"""
import argparse, collections, json, os, re, sys, time
import concurrent.futures as cf
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
from engram.core import read_rows
from engram.edit import plan, unembed_p
from engram.opt import RowOptimizer, TRIGRAM_HEADS, heads_for_step
from engram.template import trigger

ROOT = os.path.join(os.path.dirname(__file__), "..")
CODE = re.compile(r"([0-9][0-9A-Z]{5})")


def wrong_now(rec):
    n, c = rec["회사명"], rec["종목코드"]
    try:
        g = measure.generate(trigger(n, "plain"), 8)
        m = CODE.search(g)
        return None if (m and m.group(1) == c) else (n, c)
    except Exception:
        return None


def stage_fixloop(gs, by, rounds, iters, basis):
    opts, stuck = [], 0
    for r in range(rounds):
        m = gs.matched()
        if m >= gs.n_ans:
            break
        step = m if m in by else min(by)
        s = by[step]
        e0 = (read_rows(s["addrs"]).reshape(-1) if step < 0
              else unembed_p(s["target_id"]))
        o = RowOptimizer(s["addrs"], e0, gs.staged_objective(),
                         n_basis=basis, seed=31 * r + step, heads=TRIGRAM_HEADS)
        o.run(n_steps=iters, verbose=False)
        opts.append(o)
        if gs.matched() <= m:
            stuck += 1
            if stuck >= 3:
                break
        else:
            stuck = 0
    return opts


def stage_allwindow(gs, by, iters, basis):
    opts = []
    for step in sorted(by):
        s = by[step]
        e0 = (read_rows(s["addrs"]).reshape(-1) if step < 0
              else unembed_p(s["target_id"]))
        o = RowOptimizer(s["addrs"], e0, gs.staged_objective(),
                         n_basis=basis, seed=1000 + step,
                         heads=heads_for_step(step, "adaptive"))
        o.run(n_steps=iters, verbose=False)
        opts.append(o)
    return opts


def process(args_tuple):
    name, code, cfg = args_tuple
    trig, ans = trigger(name, "plain"), " " + code
    try:
        gs = measure.GenScorer(trig, ans)
        pl = plan(trig, ans, pre_steps=cfg["pre_steps"])
        by = {s["step"]: s for s in pl["steps"]}
        before = gs.matched()
        t0 = time.time()

        opts = stage_fixloop(gs, by, cfg["rounds"], cfg["iters"], cfg["basis"])
        stage = "fixloop"
        if not gs.hit():
            for o in opts:
                o.restore()
            opts = stage_allwindow(gs, by, cfg["iters"], cfg["basis"])
            stage = "allwindow"

        hit = gs.hit()
        if not hit:
            for o in opts:
                o.restore()
        return {"name": name, "code": code, "before": before,
                "after": gs.matched(), "n": gs.n_ans, "hit": hit,
                "stage": stage if hit else None, "sec": round(time.time() - t0)}
    except Exception as e:
        return {"name": name, "code": code, "hit": False, "error": str(e)[:80]}


def disjoint(cands, pre_steps, size):
    prep = []
    for n, c in cands:
        try:
            pl = plan(trigger(n, "plain"), " " + c, pre_steps=pre_steps)
        except Exception:
            continue
        used = set()
        for s in pl["steps"]:
            used.update(int(s["addrs"][h]) for h in TRIGRAM_HEADS)
        prep.append((n, c, used))
    out, cur, seen = [], [], set()
    for n, c, u in prep:
        if len(cur) >= size or (u & seen):
            if cur:
                out.append(cur)
            cur, seen = [], set()
        cur.append((n, c))
        seen |= u
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--max", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pre-steps", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--iters", type=int, default=45)
    ap.add_argument("--basis", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pipeline.json"))
    a = ap.parse_args()
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()
    cfg = {"pre_steps": a.pre_steps, "rounds": a.rounds,
           "iters": a.iters, "basis": a.basis}

    recs = json.load(open(os.path.join(ROOT, "data", "krx.json")))
    rng = np.random.default_rng(11)
    pool = [recs[i] for i in rng.choice(len(recs), min(a.pool, len(recs)), replace=False)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(16) as ex:
        wrong = [w for w in ex.map(wrong_now, pool) if w]
    print(f"검사 {len(pool)}개 -> 오답 {len(wrong)}개 ({100.0*len(wrong)/len(pool):.0f}%)"
          f"  {time.time()-t0:.0f}초")

    batches = disjoint(wrong[:a.max], a.pre_steps, a.batch)
    print(f"비충돌 배치 {len(batches)}개 (평균 {np.mean([len(b) for b in batches]):.1f}종목)\n")

    log, t0 = [], time.time()
    for bi, b in enumerate(batches):
        with cf.ThreadPoolExecutor(min(a.workers, len(b))) as ex:
            log += list(ex.map(process, [(n, c, cfg) for n, c in b]))
        h = sum(1 for r in log if r.get("hit"))
        el = time.time() - t0
        print(f"배치 {bi+1}/{len(batches)}  누적 {len(log)}  정답 {h} "
              f"({100.0*h/len(log):.0f}%)  {el/60:.0f}분  {len(log)/(el/3600):.0f}종목/시간")
        json.dump({"cfg": cfg, "elapsed_s": el, "runs": log},
                  open(a.out, "w"), ensure_ascii=False, indent=2)
    h = sum(1 for r in log if r.get("hit"))
    print(f"\n최종 {h}/{len(log)} = {100.0*h/max(1,len(log)):.1f}%")


if __name__ == "__main__":
    main()
