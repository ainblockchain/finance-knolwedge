"""독립 window 선택 + 동시 수렴 편집.

window 마다 그 행을 함께 읽는 다른 종목 수를 세어(KRX 전수 역인덱스) 독립적인
것만 고른다.  측정해 보면 직관과 다르다 — 뒷자리 숫자 window 가 오히려
독립적이고(특정 3자리 조합이 드물다), 전 종목 공통 접두('종목코드')와
이름 끝 bigram 이 최악이다.

고른 window 들은 **동시에** 최적화한다.  하나씩 고치면 뒤를 고치며 앞이
깨진다.  평가는 LLM 생성 결과 1회로 전체 조합을 본다.
"""
import argparse, collections, json, os, pickle, sys, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
from engram.core import read_rows
from engram.edit import plan, unembed_p
from engram.opt import JointOptimizer, heads_for_step, TRIGRAM_HEADS, ALL_HEADS
from engram.template import trigger

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "row_usage.pkl")


def row_usage(pre_steps=3, heads_mode="3gram"):
    """행 -> 그 행을 읽는 KRX 종목 수."""
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    krx = json.load(open(os.path.join(ROOT, "data", "krx.json")))
    idx = collections.Counter()
    for r in krx:
        try:
            pl = plan(trigger(r["회사명"], "plain"), " " + r["종목코드"],
                      pre_steps=pre_steps)
        except Exception:
            continue
        for s in pl["steps"]:
            for h in (TRIGRAM_HEADS if heads_mode == "3gram" else ALL_HEADS):
                idx[s["addrs"][h]] += 1
    pickle.dump(idx, open(CACHE, "wb"))
    return idx


def rank_windows(name, code, idx, pre_steps=3, heads=TRIGRAM_HEADS):
    pl = plan(trigger(name, "plain"), " " + code, pre_steps=pre_steps)
    out = []
    for s in pl["steps"]:
        share = sum(idx.get(s["addrs"][h], 1) - 1 for h in heads)
        out.append({"step": s["step"], "share": share, "addrs": s["addrs"],
                    "target_id": s["target_id"],
                    "ctx": "".join(reversed(s["ctx"]))})
    out.sort(key=lambda d: d["share"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--top", type=int, default=6, help="독립 상위 몇 개 window")
    ap.add_argument("--pre-steps", type=int, default=3)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--basis", type=int, default=10)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()

    idx = row_usage(a.pre_steps)
    ranked = rank_windows(a.name, a.code, idx, a.pre_steps)
    chosen = ranked[:a.top]
    trig, ans = trigger(a.name, "plain"), " " + a.code
    gs = measure.GenScorer(trig, ans)

    print(f"=== {a.name} {a.code} ===")
    print(f"편집 전 생성 {measure.generate(trig,10)!r}  일치 {gs.matched()}/{gs.n_ans}")
    print("선택한 window (독립 순):")
    for d in chosen:
        print(f"  step {d['step']:>3}  {d['ctx']!r:16s} 공유 {d['share']:6d}")
    skipped = [d["step"] for d in ranked[a.top:]]
    print(f"  제외: {skipped}\n")

    sites = []
    for d in chosen:
        e0 = (read_rows(d["addrs"]).reshape(-1) if d["step"] < 0
              else unembed_p(d["target_id"]))
        sites.append((d["addrs"], e0, TRIGRAM_HEADS))
    opt = JointOptimizer(sites, gs.staged_objective(),
                         n_basis=a.basis, seed=7)
    res = opt.run(n_steps=a.iters, log_every=max(1, a.iters // 4))

    gen = measure.generate(trig, 10)
    hit = a.code in gen
    print(f"\n편집 후 생성 {gen!r}   정답={hit}   일치 {gs.matched()}/{gs.n_ans}")
    if a.out:
        json.dump({"name": a.name, "code": a.code, "hit": hit, "gen": gen,
                   "chosen": [d["step"] for d in chosen],
                   "matched": gs.matched(), **res},
                  open(a.out, "w"), ensure_ascii=False, indent=2)
    if not a.keep:
        print(f"복원 {opt.restore()}행")


if __name__ == "__main__":
    main()
