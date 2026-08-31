"""자연 질문 형식 + 이름 자리 동시 편집 — 유일하게 6자리를 완성한 구성.

'{name}의 종목코드는' 에서 답 직전 자리는 2,804개 종목이 공유하지만
**회사명 토큰 자리는 회사 고유**다.  그 자리들만 동시 최적화하면 어텐션을 통해
숫자 자리까지 밀려 여섯 자리가 함께 맞는다 (HLB생명과학: 3/7 -> 7/7, 11.3 nat).

공유 자리를 건드리지 않으므로 종목 간 간섭도 원리적으로 적다.
"""
import argparse, json, os, re, sys, time
import concurrent.futures as cf
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
from engram.core import encode, addresses, read_rows
from engram.opt import JointOptimizer, TRIGRAM_HEADS

ROOT = os.path.join(os.path.dirname(__file__), "..")
CODE = re.compile(r"([0-9][0-9A-Z]{5})")
FMT = "{name}의 종목코드는"


def name_sites(name, code, back=5):
    """회사명 토큰 자리 window (공유 자리 제외)."""
    t = FMT.format(name=name)
    ids = encode(t + " " + code)
    p = len(encode(t))
    out = []
    for k in range(2, p - 1):          # p-1 은 ' 종목코드는' 공유 자리
        if p - 1 - k > back:
            continue
        ad = addresses(ids, k)
        sel = np.asarray(ad)[TRIGRAM_HEADS]
        out.append((ad, read_rows(sel).reshape(-1), TRIGRAM_HEADS))
    return out


def process(job):
    name, code, cfg = job
    trig, ans = FMT.format(name=name), " " + code
    try:
        gs = measure.GenScorer(trig, ans)
        before = gs.matched()
        if gs.hit():
            return {"name": name, "code": code, "skip": True, "hit": True}
        sites = name_sites(name, code, cfg["back"])
        if not sites:
            return {"name": name, "code": code, "hit": False, "why": "편집 자리 없음"}
        jo = JointOptimizer(sites, gs.staged_objective(),
                            n_basis=cfg["basis"], seed=11)
        if not jo.check_identity():
            return {"name": name, "code": code, "hit": False, "why": "시작점 불일치"}
        jo.run(n_steps=cfg["iters"], verbose=False)
        hit = gs.hit()
        after = gs.matched()
        if not hit:
            jo.restore()
        return {"name": name, "code": code, "before": before, "after": after,
                "n": gs.n_ans, "hit": hit, "sites": len(sites)}
    except Exception as e:
        return {"name": name, "code": code, "hit": False, "error": str(e)[:70]}


def wrong(rec):
    n, c = rec["회사명"], rec["종목코드"]
    try:
        g = measure.generate(FMT.format(name=n), 8)
        m = CODE.search(g)
        return None if (m and m.group(1) == c) else (n, c)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--basis", type=int, default=24)
    ap.add_argument("--back", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "natural-batch.json"))
    a = ap.parse_args()
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()
    cfg = {"iters": a.iters, "basis": a.basis, "back": a.back}

    recs = json.load(open(os.path.join(ROOT, "data", "krx.json")))
    rng = np.random.default_rng(19)
    pool = [recs[i] for i in rng.choice(len(recs), min(a.pool, len(recs)), replace=False)]
    with cf.ThreadPoolExecutor(12) as ex:
        w = [x for x in ex.map(wrong, pool) if x]
    print(f"검사 {len(pool)} -> 오답 {len(w)} ({100.0*len(w)/len(pool):.0f}%)")
    targets = w[:a.max]

    log, t0 = [], time.time()
    for i in range(0, len(targets), a.workers):
        chunk = targets[i:i + a.workers]
        with cf.ThreadPoolExecutor(len(chunk)) as ex:
            log += list(ex.map(process, [(n, c, cfg) for n, c in chunk]))
        h = sum(1 for r in log if r.get("hit"))
        print(f"  {len(log)}/{len(targets)}  정답 {h} ({100.0*h/len(log):.0f}%)  "
              f"{(time.time()-t0)/60:.0f}분", flush=True)
        json.dump({"cfg": cfg, "runs": log}, open(a.out, "w"), ensure_ascii=False, indent=2)

    h = sum(1 for r in log if r.get("hit"))
    print(f"\n최종 {h}/{len(log)} = {100.0*h/max(1,len(log)):.0f}%")
    for r in log:
        if r.get("hit") and not r.get("skip"):
            print(f"  성공 {r['name']} {r['code']}  {r['before']}/{r['n']} -> {r['after']}/{r['n']}")


if __name__ == "__main__":
    main()
