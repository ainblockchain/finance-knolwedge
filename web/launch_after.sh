#!/usr/bin/env bash

# .env 자동 로드 (SUDO_PW 등)
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/.env"
[ -f "$ENV_FILE" ] && set -a && . "$ENV_FILE" && set +a
[ -f "./.env" ] && set -a && . "./.env" && set +a
# 학습 완료 후: 학습 컨테이너 내리고 → GPU 4,5 에 "학습 후" vLLM(port 8001, TP=2) 띄우고 → 패치 끼우기.
#   web/launch_after.sh results/train-all/rows-epN.npz
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); NPZ=${1:?npz 경로}
S() { echo "${SUDO_PW:?export SUDO_PW before running}" | sudo -S -p '' "$@"; }
if S docker ps --format '{{.Names}}' | grep -q '^flashtrain$'; then
  if S docker exec flashtrain pgrep -f train_all.py >/dev/null 2>&1; then echo "학습이 아직 실행 중"; exit 1; fi
  S docker rm -f flashtrain >/dev/null; echo "학습 컨테이너 내림"
fi
# 40GB × 2 장이라 여유가 없다: 컨텍스트 8K, 동시 8, CUDA graph 는 작은 배치만 캡처, MTP 켬 (속도)
NAME=flashnext-after PORT=8001 GPUS='"device=4,5"' TP=2 MAXLEN=8192 MAXSEQS=8 GPUUTIL=0.95 COMPILE_CFG='{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8]}' PATCH_DIR="$HERE/ple_patch_after" \
  ENGRAM_HOOK=1 MTP=0 "$HERE/serve.sh"   # MTP 헤드(비양자화 MoE 5GB)는 40GB×2 에 안 들어간다
until curl -s -m 5 http://localhost:8001/v1/models >/dev/null 2>&1; do sleep 15; done; echo "학습 후 서버 준비 (8001)"
cd "$HERE"; ENGRAM_PATCH_DIR="$HERE/ple_patch_after" ENGRAM_API=http://localhost:8001 python3 scripts/patch.py apply "$NPZ"
ENGRAM_PATCH_DIR="$HERE/ple_patch_after" ENGRAM_API=http://localhost:8001 python3 scripts/patch.py status "$NPZ"
# 재기동 시 자동 재적용 감시자
nohup env ENGRAM_PATCH_DIR="$HERE/ple_patch_after" ENGRAM_API=http://localhost:8001 python3 scripts/patch_watchdog.py "$NPZ" > "$HERE/web/watchdog.log" 2>&1 &
echo "패치 감시자 시작 (web/watchdog.log)"
echo "데모: http://$(hostname -I | awk '{print $1}'):8080"
