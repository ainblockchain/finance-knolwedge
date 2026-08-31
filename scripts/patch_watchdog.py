"""vLLM 이 재기동되면(테이블이 디스크 원본으로 돌아감) 패치를 자동으로 다시 끼운다.
  ENGRAM_API=http://localhost:8001 ENGRAM_PATCH_DIR=ple_patch_after python3 scripts/patch_watchdog.py rows.npz
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engram import live
npz = sys.argv[1]; d = np.load(npz); addrs, after = d["addrs"], d["after"]
idx = np.random.default_rng(0).choice(len(addrs), 64, replace=False); sa, sv = addrs[idx], after[idx].astype(np.float32)
def applied():
    try: cur = live.read(sa).astype(np.float32); return np.abs(cur - sv).max() < 1e-3
    except Exception: return None
def apply():
    for i in range(0, len(addrs), 20000): live.write(addrs[i:i+20000], after[i:i+20000])
print(f"감시 시작: {os.path.basename(npz)} 행 {len(addrs)}", flush=True)
while True:
    st = applied()
    if st is False:
        t = time.strftime("%H:%M:%S"); apply(); print(f"[{t}] 패치 사라짐(재기동?) → 재적용 완료", flush=True)
    elif st is None:
        print(f"[{time.strftime('%H:%M:%S')}] 훅 응답 없음 (재기동 중?)", flush=True); time.sleep(30)
    time.sleep(15)
