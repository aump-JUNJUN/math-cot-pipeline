#!/usr/bin/env bash
# 多 run 对比报告：读 compare.yaml 中的 metrics JSON，写 JSON + 可选 PNG
#
# 默认读 configs/compare.yaml（不跑 infer / split / run_metrics）
# 产出：
#   reports/metrics/compare_all.json
#   reports/metrics/compare_all.png（plot_file: null 或 --no-plot 可跳过）
#
# 用法：
#   ./scripts/run_report.sh
#   ./scripts/run_report.sh --no-plot
#   CONFIG=configs/compare.yaml ./scripts/run_report.sh
#
# 前置：各 run 已跑完 run_metrics（见 run_eval_ft.sh / run_eval_base.sh）
# 依赖：pip install matplotlib（完整 eval 栈不必全装齐也能出 JSON）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "$SCRIPT_DIR/_common.sh"   # 切项目根 + 激活 conda

CONFIG="${CONFIG:-configs/compare.yaml}"

echo "==> report ($CONFIG)"
python -m src.eval.report --config "$CONFIG" "$@"

echo "report pipeline done."
