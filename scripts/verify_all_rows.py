"""학습된 PLE 행 전체를 vLLM 에 써 넣고 전 종목을 자유 생성으로 검증 (그 뒤 복원)."""
import argparse, json, os, re, sys, time
import concurrent.futures as cf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live, measure
CODE = re.compile(r"([0-9][0-9A-Z]{5})")

def run(recs, fmt):
    def one(r):
        try:
            g = measure.generate(fmt.format(name=r["회사명"]), 8); m = CODE.search(g)
            return r["회사명"], r["종목코드"], (m.group(1) if m else None), g
        except Exception as e:
            return r["회사명"], r["종목코드"], None, f"ERR {e}"
    with cf.ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, recs))

ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--fmt", default="종목코드 {name} ")
ap.add_argument("--keep", action="store_true"); ap.add_argument("--out", default="results/verify-all.json"); a = ap.parse_args()
recs = json.load(open("data/krx.json")); seen = set(); recs = [r for r in recs if not (r["종목코드"] in seen or seen.add(r["종목코드"]))]
d = np.load(a.npz); addrs, before, after = d["addrs"], d["before"], d["after"]
measure.warmup(); t0 = time.time()
print(f"종목 {len(recs)}  행 {len(addrs)}")
base = run(recs, a.fmt); b = sum(x[1] == x[2] for x in base)
print(f"편집 전: {b}/{len(recs)} = {100*b/len(recs):.1f}%  ({time.time()-t0:.0f}s)", flush=True)
def apply():
    for i in range(0, len(addrs), 20000): live.write(addrs[i:i+20000], after[i:i+20000])
sidx = np.random.default_rng(0).choice(len(addrs), 64, replace=False)
def applied():
    try: return np.abs(live.read(addrs[sidx]).astype(np.float32) - after[sidx].astype(np.float32)).max() < 1e-3
    except Exception: return False
apply(); res = []; restarts = 0
# 서버가 재기동되면 테이블이 원본으로 돌아가므로, 구간마다 확인하고 되돌아갔으면 재적용 후 그 구간을 다시 잰다
CH = 100
for i in range(0, len(recs), CH):
    while True:
        if not applied(): restarts += 1; print(f"  구간 {i}: 패치 사라짐 → 재적용 (누적 재기동 {restarts})", flush=True); apply()
        part = run(recs[i:i+CH], a.fmt)
        if applied(): res += part; break
        print(f"  구간 {i}: 측정 중 재기동 감지 → 재측정", flush=True)
h = sum(x[1] == x[2] for x in res); print(f"재기동 감지 {restarts}회")
print(f"편집 후: {h}/{len(recs)} = {100*h/len(recs):.2f}%  ({time.time()-t0:.0f}s)", flush=True)
bad = [x for x in res if x[1] != x[2]]
print("오답 예:", [(x[0], x[1], x[3]) for x in bad[:15]])
for p in ["The capital of France is", "Alexander the Great was born in", "대한민국의 수도는"]:
    print("  일반 프롬프트:", repr(p), "->", repr(measure.generate(p, 6)))
per = []
for r, x, y in zip(recs, base, res):
    per.append({"name": r["회사명"], "code": r["종목코드"], "market": r.get("시장구분", ""), "sector": r.get("업종", ""),
                "listed": r.get("상장일", ""), "before_gen": x[3], "before_ok": x[1] == x[2], "after_gen": y[3], "after_ok": y[1] == y[2]})
json.dump({"fmt": a.fmt, "n": len(recs), "before": b, "after": h, "npz": os.path.basename(a.npz), "per_stock": per},
          open(a.out, "w"), ensure_ascii=False, indent=1)
if not a.keep:
    for i in range(0, len(addrs), 20000): live.write(addrs[i:i+20000], before[i:i+20000])
    print("복원 완료")
