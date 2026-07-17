#!/usr/bin/env bash
# 训练冒烟：format → sft --smoke（GPU，约十来个 step）
#
# 默认读 configs/train.yaml 的 smoke 块，产物通常在 outputs/lora-smoke/
# 不跑 export，不污染正式 outputs/lora/
#
# 用法：
#   ./scripts/run_train_smoke.sh
#   CONFIG=configs/train.yaml ./scripts/run_train_smoke.sh
#
# 若已有 data/processed/train_messages.jsonl，可注释掉 format 一步以节省时间
# 依赖：pip install -r requirements/train.txt（含 torch / ms-swift）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "$SCRIPT_DIR/_common.sh"   # 切项目根 + 激活 conda

CONFIG="${CONFIG:-configs/train.yaml}"

echo "==> format ($CONFIG)"
python -m src.train.format --config "$CONFIG"

echo "==> sft --smoke ($CONFIG)"
python -m src.train.sft --config "$CONFIG" --smoke

echo "train smoke done."
