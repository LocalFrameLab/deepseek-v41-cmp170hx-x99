#!/usr/bin/env bash
set -euo pipefail
# Run inside a prepared vLLM environment; no container or host is selected here.
: "${VLLM_ROOT:?Set VLLM_ROOT to the patched vllm-backport checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the local model directory}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON to the prepared environment Python}"
cd "$VLLM_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset DSV41_FORCE_LOCAL_CACHE
export DSV41_HOST_STAGING_CACHE=1
export NCCL_P2P_DISABLE=1
export NCCL_PROTO=Simple
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/dsv41-cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/dsv41-triton-cache}
export HF_HUB_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export CUDA_DEVICE_MAX_CONNECTIONS=16
exec "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
  --served-model-name DeepSeek-V4.1-Flash-1017 \
  --host 127.0.0.1 --port 18050 \
  --pipeline-parallel-size 4 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.97 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --max-model-len 65536 \
  --kv-cache-tokens 131072 \
  --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"reasoning_effort":50}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3,"enable_adaptive_verification":true}' \
  --additional-config '{"deepseek_v41_hybrid":{"pipeline_layers":[6,6,6,22],"cpu_threads":[32,28],"cache_decode_interval":64,"engram_cache_gib":4,"host_cache_gib":8,"engram_storage":"ssd","route_tail_skip":0}}'
