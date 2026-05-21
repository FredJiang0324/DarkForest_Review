#!/usr/bin/env bash
set -euo pipefail

# Example only. Point these variables to model IDs or local model directories
# available in your review environment.
: "${QWEN_MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${CODER_MODEL:=Qwen/Qwen2.5-Coder-7B-Instruct}"
: "${MATHSTRAL_MODEL:=mistralai/Mathstral-7B-v0.1}"

python -m vllm.entrypoints.openai.api_server \
  --model "${QWEN_MODEL}" \
  --served-model-name qwen \
  --port 8000

# Start additional servers in separate shells, for example:
# python -m vllm.entrypoints.openai.api_server --model "${CODER_MODEL}" --served-model-name qwen_coder --port 8001
# python -m vllm.entrypoints.openai.api_server --model "${MATHSTRAL_MODEL}" --served-model-name mathstral --port 8002
