"""학습된 PLE 행 패치를 서빙 중인 vLLM 에 끼우고 뺀다 (재기동 없음).

  python3 scripts/patch.py apply  results/train-all/rows-ep5.npz     # 학습값 써 넣기
  python3 scripts/patch.py remove results/train-all/rows-ep5.npz     # 원본값으로 되돌리기
  python3 scripts/patch.py status results/train-all/rows-ep5.npz     # 현재 테이블이 어느 쪽인지
  python3 scripts/patch.py info   results/train-all/rows-ep5.npz
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from engram import live

CHUNK = 20000

def _write(addrs, rows):
    for i in range(0, len(addrs), CHUNK): live.write(addrs[i:i+CHUNK], rows[i:i+CHUNK])

def _read(addrs):
    return np.concatenate([live.read(addrs[i:i+CHUNK]) for i in range(0, len(addrs), CHUNK)])

cmd, path = sys.argv[1], sys.argv[2]
d = np.load(path); addrs, before, after = d["addrs"], d["before"], d["after"]
if cmd == "info":
    diff = np.linalg.norm(after - before, axis=1)
    print(f"행 {len(addrs)}  bf16 크기 {len(addrs)*160*2/1e6:.0f} MB  평균 변화 {diff.mean():.4f}  최대 {diff.max():.3f}")
    sys.exit()
if not live.available(): sys.exit("패치 훅 없음 (ENGRAM_HOOK=1 로 서빙 필요)")
t0 = time.time()
if cmd == "apply":   _write(addrs, after);  print(f"끼움: {len(addrs)}행  {time.time()-t0:.1f}s")
elif cmd == "remove": _write(addrs, before); print(f"뺌: {len(addrs)}행  {time.time()-t0:.1f}s")
elif cmd == "status":
    idx = np.random.default_rng(0).choice(len(addrs), min(2000, len(addrs)), replace=False)
    cur = _read(addrs[idx]).astype(np.float32)
    da = np.abs(cur - after[idx].astype(np.float32)).max(); db = np.abs(cur - before[idx].astype(np.float32)).max()
    print("현재 테이블:", "학습값(끼워짐)" if da < db else "원본값(빠짐)", f"  |학습값 차 {da:.4f}  원본 차 {db:.4f}  (표본 {len(idx)}행)")
else: sys.exit("apply|remove|status|info")
