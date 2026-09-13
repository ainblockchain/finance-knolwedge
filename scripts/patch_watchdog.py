"""vLLM 이 재기동되면(테이블이 디스크 원본으로 돌아감) 패치를 자동으로 다시 끼운다.

  ENGRAM_API=http://localhost:8001 ENGRAM_PATCH_DIR=ple_patch_after python3 scripts/patch_watchdog.py rows.npz

2026-09-12 사고: 11:39 에 컨테이너가 재기동하자 훅이 잠깐 무응답이었고, 그 뒤로 이 감시자는
**22시간 동안 아무 로그도 남기지 않은 채 재적용을 하지 않았다.** 그 사이 데모의 "학습 후" 패널은
원본 모델을 서빙했고, 화면에는 아무 이상도 표시되지 않았다.

이전 판이 그렇게 될 수 있었던 자리가 셋이다. 정상 경로에 로그가 없어서 멈춘 것과 도는 것이
구분되지 않았고(`st is True` 는 아무것도 찍지 않는다), `apply()` 가 예외를 막지 않아 재기동 창에
걸린 write 하나가 루프 전체를 끝낼 수 있었고, "재적용 완료" 를 실제로 들어갔는지 확인하지 않고
찍었다. 어느 자리였는지는 사후에 특정하지 못했으므로 셋 다 닫는다.

  1. **조용하면 죽은 것이다.** 정상일 때도 HEARTBEAT_S 마다 한 줄 남긴다. 마지막 줄의 시각이
     곧 이 감시자가 마지막으로 살아 있던 시각이고, 로그가 멈췄다는 사실 자체가 신호가 된다.
  2. **어떤 예외도 루프를 멈추지 못한다.** 훅 무응답이든 무엇이든 남기고 계속 돈다.
  3. **재적용은 검증까지가 한 세트다.** 끼운 뒤 다시 읽어 확인하고, 확인될 때까지 재시도한다.
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from engram import live
from patch_fingerprint import check, load

POLL_S = float(os.environ.get("WATCHDOG_POLL_S", "15"))
HEARTBEAT_S = float(os.environ.get("WATCHDOG_HEARTBEAT_S", "600"))
APPLY_RETRY = int(os.environ.get("WATCHDOG_APPLY_RETRY", "3"))
CHUNK = 20000


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def apply_all(addrs, after):
    for i in range(0, len(addrs), CHUNK):
        live.write(addrs[i:i + CHUNK], after[i:i + CHUNK])


def main():
    npz = sys.argv[1]
    d = np.load(npz)
    addrs, after = d["addrs"], d["after"]
    fp = load(npz)
    patch_dir = os.environ.get("ENGRAM_PATCH_DIR")
    log(f"감시 시작: {os.path.basename(npz)} 행 {len(addrs)} · 폴링 {POLL_S:.0f}s · 하트비트 {HEARTBEAT_S:.0f}s")

    last_state, last_beat = None, 0.0
    while True:
        try:
            state, detail = check(fp, patch_dir)
            # 상태가 바뀐 순간은 반드시 남기고, 안 바뀌어도 하트비트 주기마다 남긴다 (규칙 1).
            if state != last_state or time.time() - last_beat >= HEARTBEAT_S:
                log(f"{state} — {detail}")
                last_beat = time.time()
            last_state = state

            if state == "stale":
                for attempt in range(1, APPLY_RETRY + 1):
                    t0 = time.time()
                    apply_all(addrs, after)
                    again, detail2 = check(fp, patch_dir)          # 규칙 3: 확인하고 나서 찍는다
                    if again == "applied":
                        log(f"재적용 완료 · {len(addrs)}행 {time.time() - t0:.1f}s · 검증 {detail2}")
                        last_state, last_beat = "applied", time.time()
                        break
                    log(f"재적용 {attempt}/{APPLY_RETRY} 실패 — 검증 결과 {again} ({detail2})")
                    time.sleep(POLL_S)
                else:
                    log("재적용이 계속 실패한다. 훅과 컨테이너 로그를 확인할 것 — 모델은 원본 상태다")
                    last_state = None
        except Exception as e:  # 루프는 무슨 일이 있어도 멈추지 않는다 (규칙 2)
            log(f"점검 중 예외 — {type(e).__name__}: {e}")
            last_state, last_beat = None, time.time()
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
