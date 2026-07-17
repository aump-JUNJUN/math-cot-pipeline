#!/usr/bin/env bash
# 正式训练链：format → sft → export（GPU）
#
# 默认读 configs/train.yaml
# 产出：
#   outputs/lora/checkpoint-*
#   outputs/merged/best/
#
# 用法：
#   ./scripts/run_train.sh
#   CONFIG=configs/train.yaml ./scripts/run_train.sh
#
# 依赖：pip install -r requirements/train.txt（含 torch / ms-swift）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

CONFIG="${CONFIG:-configs/train.yaml}"

echo "==> format ($CONFIG)"
python -m src.train.format --config "$CONFIG"

echo "==> sft ($CONFIG)"
python -m src.train.sft --config "$CONFIG"

echo "==> export ($CONFIG)"
python -m src.train.export --config "$CONFIG"

echo "train pipeline done."

