from __future__ import annotations

"""
评测流水线第 2 步：拆分 generated_text（CPU，无需 GPU）。

读 configs/eval_ft.yaml 或 eval_base.yaml → split 块 + reports/predictions/{name}.jsonl，
对每条 generated_text 调用 common/extract.split_solution()，
分别写入 answer_file 与 cot_file。

在整条 eval 链路中的位置：
  infer.py       → reports/predictions/{name}.jsonl
  split.py       → reports/predictions/{name}_answers.jsonl   （本文件）
                   reports/predictions/{name}_cots.jsonl
  run_metrics.py → reports/metrics/{name}_answer.json
                   reports/metrics/{name}_cot.json
  report.py      → reports/metrics/compare_all.json + .png   （读 configs/compare.yaml）

可反复运行；改 split.strategies 或 cot_mode 不必重跑 infer。

启动：
  python -m src.eval.split
  python -m src.eval.split --config configs/eval_base.yaml
"""

import argparse
from pathlib import Path
from typing import Any

from src.common.config import load_config
from src.common.extract import split_solution
from src.common.io import read_jsonl, write_jsonl


def _apply_cot_mode(generated_text: str, cot: str, cot_mode: str) -> str:
    """
    决定 pred_cot 写什么：
      before_answer — split_solution 抽出的 COT（\\boxed 之前）
      full_text     — 整段 generated_text
    """
    if cot_mode == "full_text":
        return generated_text.strip()
    if cot_mode == "before_answer":
        return cot
    raise ValueError(
        f"Unsupported split.cot_mode: {cot_mode}. "
        "Choose from: before_answer, full_text."
    )


def _split_one_row(
    row: dict[str, Any],
    strategies: list[str],
    cot_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """拆分单行 prediction，返回 (answer_row, cot_row)。"""
    generated_text = row.get("generated_text")
    if generated_text is None:
        raise KeyError(
            f"Missing 'generated_text' in prediction row id={row.get('id')!r}. "
            "Re-run infer or check prediction_file."
        )

    text = str(generated_text)
    result = split_solution(text, strategies)
    pred_cot = _apply_cot_mode(text, result.cot, cot_mode)

    answer_row = {
        "id": row.get("id"),
        "problem": row.get("problem"),
        "pred_answer": result.answer,
        "extract_ok": result.extract_ok,
        "strategy": result.strategy,
    }
    cot_row = {
        "id": row.get("id"),
        "problem": row.get("problem"),
        "pred_cot": pred_cot,
        "extract_ok": result.extract_ok,
    }
    return answer_row, cot_row


def run_split_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """
    主流程：读 predictions → 批量 split → 写 answer_file / cot_file。

    可反复运行；改 strategies 或 cot_mode 不必重跑 infer。
    """
    split_cfg = config["split"]

    prediction_path = Path(split_cfg["prediction_file"])
    answer_path = Path(split_cfg["answer_file"])
    cot_path = Path(split_cfg["cot_file"])
    strategies = list(split_cfg.get("strategies", ["boxed"]))
    cot_mode = split_cfg.get("cot_mode", "before_answer")

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_path}. "
            "Run `python -m src.eval.infer --config <same eval yaml>` first."
        )

    answer_rows: list[dict[str, Any]] = []
    cot_rows: list[dict[str, Any]] = []
    extract_ok_count = 0

    for row in read_jsonl(prediction_path):
        answer_row, cot_row = _split_one_row(row, strategies, cot_mode)
        answer_rows.append(answer_row)
        cot_rows.append(cot_row)
        if answer_row["extract_ok"]:
            extract_ok_count += 1

    answer_written = write_jsonl(answer_path, answer_rows)
    cot_written = write_jsonl(cot_path, cot_rows)

    total = len(answer_rows)
    extract_rate = (extract_ok_count / total) if total else 0.0

    return {
        "prediction_file": str(prediction_path),
        "answer_file": str(answer_path),
        "cot_file": str(cot_path),
        "rows": total,
        "answer_rows_written": answer_written,
        "cot_rows_written": cot_written,
        "extract_ok_count": extract_ok_count,
        "extract_rate": extract_rate,
        "strategies": strategies,
        "cot_mode": cot_mode,
    }


def run_from_config(config_path: str | Path) -> dict[str, Any]:
    """从 yaml 路径加载 config，再调用 run_split_pipeline。"""
    config = load_config(config_path)
    return run_split_pipeline(config)


def main() -> None:
    """CLI 入口：python -m src.eval.split"""
    parser = argparse.ArgumentParser(description="Split generated_text into pred_answer / pred_cot")
    parser.add_argument("--config", default="configs/eval_ft.yaml", help="评测配置文件路径")
    args = parser.parse_args()

    result = run_from_config(args.config)
    print("split done:")
    print(f"  prediction_file: {result['prediction_file']}")
    print(f"  answer_file: {result['answer_file']}")
    print(f"  cot_file: {result['cot_file']}")
    print(f"  rows: {result['rows']}")
    print(f"  extract_rate: {result['extract_rate']:.4f}")


if __name__ == "__main__":
    main()