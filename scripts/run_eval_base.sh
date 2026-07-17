#!/usr/bin/env bash
# 基座模型评测链：infer → split → run_metrics
#
# 默认读 configs/eval_base.yaml（Hub 权重，model_path: null）
# 产出：
#   reports/predictions/base.jsonl / base_answers.jsonl / base_cots.jsonl
#   reports/metrics/base_answer.json / base_cot.json
#
# 用法：
#   ./scripts/run_eval_base.sh
#   ./scripts/run_eval_base.sh --limit 4          # 试跑 infer，参数透传
#   TASK=answer ./scripts/run_eval_base.sh        # 只跑 answer 侧 metrics
#   CONFIG=configs/eval_base.yaml ./scripts/run_eval_base.sh
#
# 依赖：GPU（infer）+ pip install -r requirements/eval.txt
# 对比报告另跑：./scripts/run_report.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "$SCRIPT_DIR/_common.sh"   # 切项目根 + 激活 conda

CONFIG="${CONFIG:-configs/eval_base.yaml}"
TASK="${TASK:-all}"               # answer | cot | all

echo "==> infer ($CONFIG)"
python -m src.eval.infer --config "$CONFIG" "$@"

echo "==> split ($CONFIG)"
python -m src.eval.split --config "$CONFIG"

echo "==> run_metrics ($CONFIG, task=$TASK)"
python -m src.eval.run_metrics --config "$CONFIG" --task "$TASK"

echo "eval base pipeline done."
