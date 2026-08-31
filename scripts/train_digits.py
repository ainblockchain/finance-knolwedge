"""자릿수 행 학습 — 충돌 행은 소비자 여럿을 함께 본다.

pos 0 은 공백 토큰이라 학습 대상이 아니다.  실제 숫자는 pos 1~6 이고
충돌률이 pos1 40% -> pos4 100% 로 올라간다.

행은 공유 파라미터이므로 한 종목에 맞추면 다른 종목이 깨진다.  그래서
목적함수에 **그 행을 함께 읽는 다른 쌍**을 넣는다:

    점수 = 내 margin(cap)  +  λ · Σ min(남의 margin, 0)

남이 이미 음수면 더 나빠지는 것만 벌한다.  전수를 다 보면 비싸므로
소비자를 k개 샘플링한다 (미니배치).
"""
import argparse, json, os, pickle, random, re, sys, time
import concurrent.futures as cf
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
from engram.core import encode, addresses, tokenizer
from engram.edit import unembed_p
from engram.opt import RowOptimizer, TRIGRAM_HEADS
from engram.template import trigger

ROOT = os.path.join(os.path.dirname(__file__), "..")
CODE = re.compile(r"([0-9][0-9A-Z]{5})")
DS = pickle.load(open(os.path.join(ROOT, "data", "dataset.pkl"), "rb"))
PAIRS, ROW2CONS = DS["pairs"], DS["row2cons"]


def margin(prompt, target_s):
    r = measure._post("/v1/completions", {
        "model": measure.MODEL, "prompt": prompt,
        "max_tokens": 1, "temperature": 0, "logprobs": 5})
    tl = r["choices"][0]["logprobs"]["top_logprobs"][0] or {}
    mine = tl.get(target_s)
    other = max((v for k, v in tl.items() if k != target_s), default=None)
    top = max(tl.items(), key=lambda kv: kv[1])[0] if tl else None
    return ((mine - other) if (mine is not None and other is not None) else -10.0), top


def prompt_for(pair):
    t = trigger(pair["name"], "plain")
    return t + " " + pair["code"][:pair["pos"] - 1] if pair["pos"] > 1 else t + " "


def train_pair(job):
    """한 (종목, 자리) 쌍의 행을 학습.  충돌 소비자를 함께 본다."""
    pi, cfg = job
    p = PAIRS[pi]
    try:
        t = trigger(p["name"], "plain")
        ids = encode(t + " " + p["code"])
        base = len(encode(t)) - 1
        pos = base + p["pos"]
        a = addresses(ids, pos)
        my_prompt = tokenizer().decode(ids[:pos + 1])
        my_t = p["target_s"]

        m0, top0 = margin(my_prompt, my_t)
        if top0 == my_t:
            return {"pi": pi, "skip": True, "fixed": True}

        # 같은 행을 읽는 다른 쌍들 (충돌 소비자)
        others = set()
        for h in TRIGRAM_HEADS:
            others.update(ROW2CONS.get(int(a[h]), []))
        others.discard(pi)
        others = [PAIRS[i] for i in others if PAIRS[i]["target_s"] != my_t]
        random.shuffle(others)
        others = others[:cfg["k"]]
        oprompts = []
        for q in others:
            qi = encode(trigger(q["name"], "plain") + " " + q["code"])
            qb = len(encode(trigger(q["name"], "plain"))) - 1
            oprompts.append((tokenizer().decode(qi[:qb + q["pos"] + 1]), q["target_s"]))

        def obj():
            s = min(margin(my_prompt, my_t)[0], 1.5)
            for pr, tg in oprompts:
                s += cfg["lam"] * min(margin(pr, tg)[0], 0.0)
            return s

        o = RowOptimizer(a, unembed_p(p["target"]), obj,
                         n_basis=cfg["basis"], seed=17, heads=TRIGRAM_HEADS)
        o.run(n_steps=cfg["iters"], verbose=False)
        m1, top1 = margin(my_prompt, my_t)
        ok = (top1 == my_t)
        broke = sum(1 for pr, tg in oprompts if margin(pr, tg)[1] != tg)
        if not ok:
            o.restore()
        return {"pi": pi, "name": p["name"], "pos": p["pos"], "m0": m0, "m1": m1,
                "fixed": ok, "n_others": len(others), "broke": broke}
    except Exception as e:
        return {"pi": pi, "error": str(e)[:70], "fixed": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=1)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--basis", type=int, default=16)
    ap.add_argument("--k", type=int, default=2, help="함께 볼 충돌 소비자 수")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()
    cfg = {"iters": a.iters, "basis": a.basis, "k": a.k, "lam": a.lam}

    idxs = [i for i, p in enumerate(PAIRS) if p["pos"] == a.pos]
    rng = np.random.default_rng(3)
    idxs = [int(i) for i in rng.choice(idxs, min(a.n, len(idxs)), replace=False)]

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(train_pair, [(i, cfg) for i in idxs]))
    el = time.time() - t0
    done = [r for r in res if not r.get("error")]
    skip = sum(1 for r in done if r.get("skip"))
    fixed = sum(1 for r in done if r.get("fixed"))
    broke = sum(r.get("broke", 0) for r in done)
    print(f"pos {a.pos}  {len(done)}쌍  {el/60:.1f}분  "
          f"({len(done)/(el/3600):.0f}쌍/시간, {a.workers}병렬)")
    print(f"  이미 맞음 {skip}, 학습으로 맞춤 {fixed-skip}, 합계 {fixed} "
          f"({100.0*fixed/len(done):.0f}%)")
    print(f"  학습 중 깨뜨린 다른 쌍 {broke}")
    if a.out:
        json.dump({"cfg": cfg, "pos": a.pos, "elapsed_s": el, "runs": res},
                  open(a.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
