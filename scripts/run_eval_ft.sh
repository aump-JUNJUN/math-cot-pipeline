#!/usr/bin/env bash
# 微调模型评测链：infer → split → run_metrics
#
# 默认读 configs/eval_ft.yaml（加载 outputs/merged/best）
# 产出：
#   reports/predictions/ft.jsonl / ft_answers.jsonl / ft_cots.jsonl
#   reports/metrics/ft_answer.json / ft_cot.json
#
# 用法：
#   ./scripts/run_eval_ft.sh
#   ./scripts/run_eval_ft.sh --limit 4          # 试跑 infer，参数仅透传给 infer
#   TASK=answer ./scripts/run_eval_ft.sh        # 只跑 answer 侧 metrics
#   CONFIG=configs/eval_ft.yaml ./scripts/run_eval_ft.sh
#
# 前置：已 run_train.sh + export（outputs/merged/best 存在）
# 若 predictions 已有，可注释掉 infer 一步，只跑 split + run_metrics
# 依赖：GPU（infer）+ pip install -r requirements/eval.txt
# 对比报告另跑：./scripts/run_report.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "$SCRIPT_DIR/_common.sh"   # 切项目根 + 激活 conda

CONFIG="${CONFIG:-configs/eval_ft.yaml}"
TASK="${TASK:-all}"               # answer | cot | all

echo "==> infer ($CONFIG)"
python -m src.eval.infer --config "$CONFIG" "$@"

echo "==> split ($CONFIG)"
python -m src.eval.split --config "$CONFIG"

echo "==> run_metrics ($CONFIG, task=$TASK)"
python -m src.eval.run_metrics --config "$CONFIG" --task "$TASK"

echo "eval ft pipeline done."
