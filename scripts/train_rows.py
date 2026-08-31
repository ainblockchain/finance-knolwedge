"""행 단위 학습 — 한 행을 그 행을 읽는 **모든 쌍**에 대해 함께 최적화한다.

종목 단위 편집은 원리적으로 틀렸다.  행이 공유 파라미터이기 때문이다
(pos 4 는 쌍의 100% 가 충돌 행을 읽는다).  행마다 소비자 전체를 목적함수에 넣어야 한다.

우선순위는 데이터가 정해준다:
    pos 0  충돌  0%   <- 2,804 종목이 서로 간섭하지 않음.  완전 병렬 가능
    pos 1  충돌 40%
    pos 2  충돌 76%
    pos 3+ 충돌 99~100%

pos 0 부터 확정하고, 첫 자리가 맞으면 나머지가 따라오는지 측정한다.
"""
import argparse, collections, json, os, pickle, re, sys, time
import concurrent.futures as cf
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
from engram.core import encode, addresses
from engram.edit import unembed_p
from engram.opt import RowOptimizer, TRIGRAM_HEADS
from engram.template import trigger

ROOT = os.path.join(os.path.dirname(__file__), "..")
CODE = re.compile(r"([0-9][0-9A-Z]{5})")


class PosScorer:
    """한 위치의 목표 토큰 margin.  1토큰 생성이라 평가가 싸다."""

    def __init__(self, prompt, target_s):
        self.p, self.t = prompt, target_s

    def probe(self):
        r = measure._post("/v1/completions", {
            "model": measure.MODEL, "prompt": self.p,
            "max_tokens": 1, "temperature": 0, "logprobs": 5})
        tl = r["choices"][0]["logprobs"]["top_logprobs"][0] or {}
        mine = tl.get(self.t)
        other = max((v for k, v in tl.items() if k != self.t), default=None)
        top = max(tl.items(), key=lambda kv: kv[1])[0] if tl else None
        return ((mine - other) if (mine is not None and other is not None)
                else -10.0), top


def train_one(job):
    """pos 0 한 종목: trigger 마지막 토큰 위치의 8행을 최적화."""
    name, code, cfg = job
    try:
        t = trigger(name, "plain")
        ids = encode(t + " " + code)
        p = len(encode(t))
        pos = p - 1
        a = addresses(ids, pos)
        tgt_id = ids[pos + 1]
        sc = PosScorer(t, measure.__dict__ and __import__("engram.core", fromlist=["x"]).tokenizer().decode([tgt_id]))
        m0, top0 = sc.probe()
        if top0 == sc.t:
            return {"name": name, "code": code, "skip": "이미 맞음",
                    "m0": m0, "m1": m0, "fixed": True}
        o = RowOptimizer(a, unembed_p(tgt_id), lambda: min(sc.probe()[0], 1.5),
                         n_basis=cfg["basis"], seed=17, heads=TRIGRAM_HEADS)
        o.run(n_steps=cfg["iters"], verbose=False)
        m1, top1 = sc.probe()
        ok = (top1 == sc.t)
        if not ok:
            o.restore()
        return {"name": name, "code": code, "m0": m0, "m1": m1, "fixed": ok}
    except Exception as e:
        return {"name": name, "code": code, "error": str(e)[:70], "fixed": False}


def full_code_ok(name, code):
    g = measure.generate(trigger(name, "plain"), 8)
    m = CODE.search(g)
    return bool(m and m.group(1) == code), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--iters", type=int, default=35)
    ap.add_argument("--basis", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "train-pos0.json"))
    a = ap.parse_args()
    if not live.available():
        sys.exit("패치 훅 없음")
    measure.warmup()
    cfg = {"iters": a.iters, "basis": a.basis}

    krx = json.load(open(os.path.join(ROOT, "data", "krx.json")))
    rng = np.random.default_rng(5)
    sample = [krx[i] for i in rng.choice(len(krx), min(a.n, len(krx)), replace=False)]
    jobs = [(r["회사명"], r["종목코드"], cfg) for r in sample]

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(train_one, jobs))
    el = time.time() - t0
    fixed = sum(1 for r in res if r.get("fixed"))
    already = sum(1 for r in res if r.get("skip"))
    print(f"pos 0 학습 {len(res)}종목  {el/60:.1f}분  "
          f"({len(res)/(el/3600):.0f}종목/시간, {a.workers}병렬)")
    print(f"  첫 자리 이미 맞음 {already}, 학습으로 맞춤 {fixed-already}, "
          f"합계 {fixed} ({100.0*fixed/len(res):.0f}%)")

    # 첫 자리가 맞으면 전체 코드가 따라오는가
    with cf.ThreadPoolExecutor(a.workers) as ex:
        full = list(ex.map(lambda r: full_code_ok(r["name"], r["code"])[0],
                           [r for r in res if r.get("fixed")]))
    print(f"  그중 전체 코드까지 정답 {sum(full)} "
          f"({100.0*sum(full)/max(1,len(full)):.0f}%)")
    json.dump({"cfg": cfg, "elapsed_s": el, "runs": res,
               "full_ok": int(sum(full))}, open(a.out, "w"), ensure_ascii=False)
    print(f"기록 -> {a.out}")


if __name__ == "__main__":
    main()
