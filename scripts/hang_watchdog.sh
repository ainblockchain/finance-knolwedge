#!/usr/bin/env bash

# .env 자동 로드 (SUDO_PW 등)
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/.env"
[ -f "$ENV_FILE" ] && set -a && . "$ENV_FILE" && set +a
[ -f "./.env" ] && set -a && . "./.env" && set +a
# vLLM PLE 오프로드 멈춤 감시: 컨테이너 로그에 "No available shared memory broadcast block" 이
# 연속 2회(약 2분) 나오면 5분 RPC 타임아웃을 기다리지 않고 바로 재기동한다.
#   scripts/hang_watchdog.sh flashnext flashnext-after
S() { echo "${SUDO_PW:?export SUDO_PW before running}" | sudo -S -p '' "$@"; }
declare -A last
while true; do
  for c in "$@"; do
    n=$(S docker logs --since 150s "$c" 2>&1 | grep -c "No available shared memory broadcast block")
    if [ "$n" -ge 2 ]; then
      echo "[$(date +%H:%M:%S)] $c 멈춤 감지 ($n회) → 재기동"; S docker restart "$c" >/dev/null; sleep 240
    fi
  done
  sleep 30
done
