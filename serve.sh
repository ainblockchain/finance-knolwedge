#!/usr/bin/env bash

# Qwen3.8-Flash-Next W4A16 on A100-SXM4-40GB
# 이 호스트는 A100 8장 중 7장만 살아있다(PCI 0000:24:05.0 하드웨어 불량).
# 아키텍처 제약: TP는 24(attn heads)/16(linear k heads)/48(linear v heads)를 모두
# 나눠야 하므로 TP ∈ {1,2,4,8} 만 가능. 7장이라 최대 TP=4 (GPU 0~3).
# --enable-expert-parallel 필수: moe_intermediate_size=640 이 group-128 로 나눠떨어지지 않아
#   TP 샤딩만으로는 양자화 그룹이 깨진다.
# --disable-custom-all-reduce 필수: 이 컨테이너 환경에서 vLLM 커스텀 all-reduce 가
#   custom_all_reduce.cuh:164 'invalid argument' 로 죽는다. NVLink NV12 이므로 NCCL 로도 손실 없음.
# --compilation-config mode 0: inductor 컴파일이 Ampere 에서 무한 행. 디코드 CUDA 그래프만 사용.
#
# ENGRAM_HOOK=1 로 띄우면 PLE(Engram) 테이블 in-place 패치 훅이 함께 올라간다.
#   - vllm_patch/{worker.py,patch_hook.py,connector.py} 를 컨테이너 위에 bind-mount
#   - ple_patch/ 디렉터리를 통해 재기동 없이 테이블 행을 읽고 쓴다
#   - 클라이언트: engram/live.py
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR=${MODEL_DIR:-/mnt/newdata/models/Qwen3.8-Flash-Next-W4A16}
IMAGE=${IMAGE:-vllm/vllm-openai:qwen38-flash-next}
NAME=${NAME:-flashnext}
PORT=${PORT:-8000}
GPUS=${GPUS:-'"device=0,1,2,3"'}
TP=${TP:-4}
MAXLEN=${MAXLEN:-262144}
PATCH_DIR=${PATCH_DIR:-$HERE/ple_patch}   # 인스턴스마다 다른 디렉터리를 써야 훅이 섞이지 않는다
# MTP 추측 디코딩.  편집 실험 중에는 끈다 — 추론 도중 PLE 행이 바뀌면
# 드래프트/검증 불일치로 GPU 워커가 멈추고 EngineCore 가 죽는다 (실측).
MTP=${MTP:-1}
MAXSEQS=${MAXSEQS:-64}
GPUUTIL=${GPUUTIL:-0.90}
# 추가 vLLM 인자 (예: EXTRA_ARGS="--enforce-eager")
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
COMPILE_CFG=${COMPILE_CFG:-'{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'}
ENGRAM_HOOK=${ENGRAM_HOOK:-0}

S() { sudo "$@"; }

HOOK_ARGS=()
if [ "$ENGRAM_HOOK" = "1" ]; then
  VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
  mkdir -p "$PATCH_DIR"
  HOOK_ARGS=(
    -e VLLM_PLE_PATCH_DIR=/ple_patch
    -v "$PATCH_DIR":/ple_patch
    -v "$HERE/vllm_patch/worker.py":"$VLLM_PKG/v1/ple_offload/worker.py":ro
    -v "$HERE/vllm_patch/patch_hook.py":"$VLLM_PKG/v1/ple_offload/patch_hook.py":ro
    -v "$HERE/vllm_patch/connector.py":"$VLLM_PKG/v1/ple_offload/connector.py":ro
  )
  echo "engram patch hook: ENABLED  (patch dir $PATCH_DIR)"
fi

SPEC_ARGS=()
if [ "$MTP" = "1" ]; then
  SPEC_ARGS=(--speculative-config '{"method": "mtp", "num_speculative_tokens": 4}')
else
  echo "MTP: 비활성 (편집 실험 모드)"
fi

S docker rm -f "$NAME" >/dev/null 2>&1 || true

S docker run -d --name "$NAME" \
  --gpus "$GPUS" --shm-size 32g --ipc=host \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --restart unless-stopped \
  -v "$MODEL_DIR":/model:ro \
  "${HOOK_ARGS[@]}" \
  -p 127.0.0.1:${PORT}:8000 \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_LOGGING_LEVEL=INFO \
  "$IMAGE" \
  --model /model --served-model-name Qwen3.8-Flash-Next \
  --tensor-parallel-size "$TP" --enable-expert-parallel \
  --disable-custom-all-reduce \
  --gpu-memory-utilization "$GPUUTIL" \
  --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
  --compilation-config "$COMPILE_CFG" \
  "${SPEC_ARGS[@]}" "${EXTRA[@]}" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --trust-remote-code

echo "started: $NAME (TP=$TP, port=$PORT)"
echo "logs: sudo docker logs -f $NAME"
