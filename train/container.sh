#!/usr/bin/env bash

# .env 자동 로드 (SUDO_PW 등)
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/.env"
[ -f "$ENV_FILE" ] && set -a && . "$ENV_FILE" && set +a
[ -f "./.env" ] && set -a && . "./.env" && set +a
# 학습 컨테이너 (서빙 컨테이너 flashnext 와 별개).  GPU 4,5,6 사용.
#   train/container.sh up      컨테이너 생성 + 패키지 설치
#   train/container.sh run <cmd...>   컨테이너 안에서 실행
#   train/container.sh down
set -euo pipefail
S() { echo "${SUDO_PW:?export SUDO_PW before running}" | sudo -S -p '' "$@"; }
NAME=flashtrain
IMG=vllm/vllm-openai:qwen38-flash-next
MODEL=/mnt/newdata/models/Qwen3.8-Flash-Next-W4A16
HERE=/mnt/newdata/qwen3.8
case "${1:-}" in
  up)
    S docker rm -f $NAME >/dev/null 2>&1 || true
    S docker run -d --name $NAME --gpus '"device=4,5,6"' --ipc=host --shm-size=64g \
      -v "$MODEL":/model:ro -v "$HERE":/work -w /work \
      -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
      --entrypoint bash $IMG -c "sleep infinity"
    S docker exec $NAME bash -c "pip install -q https://github.com/huggingface/transformers/archive/refs/heads/main.zip flash-linear-attention==0.5.2 2>&1 | tail -3 && python3 -c \"import transformers;print('transformers',transformers.__version__); import fla; print('fla', fla.__version__)\""
    ;;
  run) shift; S true; sudo docker exec -i -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $NAME "$@" ;;   # sudo 인증 먼저 — stdin 을 스크립트에 넘기기 위해
  down) S docker rm -f $NAME ;;
  *) echo "usage: $0 up|run|down"; exit 1 ;;
esac
