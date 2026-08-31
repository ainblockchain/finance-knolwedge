#!/usr/bin/env bash

# .env 자동 로드 (SUDO_PW 등)
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/.env"
[ -f "$ENV_FILE" ] && set -a && . "$ENV_FILE" && set +a
[ -f "./.env" ] && set -a && . "./.env" && set +a
SUDO_PW="${SUDO_PW:?export SUDO_PW before running}"
IMG=vllm/vllm-openai:qwen38-flash-next
for i in $(seq 1 40); do
  echo "=== pull attempt $i $(date +%T) ==="
  echo "$SUDO_PW" | sudo -S -p '' docker pull "$IMG" 2>&1 | tail -3
  if echo "$SUDO_PW" | sudo -S -p '' docker images -q "$IMG" | grep -q .; then echo "PULL_OK $(date +%T)"; exit 0; fi
  sleep 20
done
echo "PULL_FAILED"; exit 1
