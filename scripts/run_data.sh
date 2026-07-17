#!/usr/bin/env bash
# 数据链：download → clean（CPU，无需 GPU）
#
# 读 configs/data.yaml，产出：
#   data/raw/*.jsonl
#   data/processed/train.jsonl / test.jsonl
#
# 用法：
#   ./scripts/run_data.sh
#   bash scripts/run_data.sh
#
# 依赖：conda activate math-cot && pip install -r requirements/base.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "$SCRIPT_DIR/_common.sh"   # 切项目根 + 激活 conda

echo "==> download (configs/data.yaml)"
python -m src.data.download

echo "==> clean (configs/data.yaml)"
python -m src.data.clean

echo "data pipeline done."
