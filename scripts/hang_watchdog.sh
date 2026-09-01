#!/usr/bin/env bash

# vLLM PLE 오프로드 멈춤 감시: 컨테이너 로그에 "No available shared memory broadcast block" 이
# 연속 2회(약 2분) 나오면 5분 RPC 타임아웃을 기다리지 않고 바로 재기동한다.
#   scripts/hang_watchdog.sh flashnext flashnext-after
S() { "$@"; }
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
