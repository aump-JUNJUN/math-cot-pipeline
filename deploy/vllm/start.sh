#!/usr/bin/env bash
# 启动 vLLM API（读 configs/serve.yaml）
#
# 本地：
#   bash deploy/vllm/start.sh
#
# Docker（compose 里挂载 /model 和 /config/serve.yaml）：
#   MODEL_PATH=/model SERVE_CONFIG=/config/serve.yaml bash deploy/vllm/start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export SERVE_CONFIG="${SERVE_CONFIG:-$ROOT/configs/serve.yaml}"
export MODEL_PATH="${MODEL_PATH:-}"

cd "$ROOT"

python3 -c "import yaml" 2>/dev/null || pip install -q -r requirements/serve.txt

exec python3 "$SCRIPT_DIR/render_cmd.py"