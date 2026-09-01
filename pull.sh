#!/usr/bin/env bash

IMG=vllm/vllm-openai:qwen38-flash-next
for i in $(seq 1 40); do
  echo "=== pull attempt $i $(date +%T) ==="
  docker pull "$IMG" 2>&1 | tail -3
  if docker images -q "$IMG" | grep -q .; then echo "PULL_OK $(date +%T)"; exit 0; fi
  sleep 20
done
echo "PULL_FAILED"; exit 1
