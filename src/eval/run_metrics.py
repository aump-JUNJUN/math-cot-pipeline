from __future__ import annotations

"""
评测流水线第 3 步：聚合 answer / cot 指标（CPU，无需 GPU）。

读 configs/eval_ft.yaml 或 eval_base.yaml → answer_metrics / cot_metrics 块，
按 join_key 对齐 pred_file 与 gold_file，调用 metrics.py 中注册的指标类，
写入 reports/metrics/{name}_answer.json 与 {name}_cot.json。

在整条 eval 链路中的位置：
  infer.py       → reports/predictions/{name}.jsonl
  split.py       → reports/predictions/{name}_answers.jsonl / _cots.jsonl
  run_metrics.py → reports/metrics/{name}_answer.json / _cot.json  （本文件）
  report.py      → reports/metrics/compare_all.json + .png   （读 configs/compare.yaml）

可反复运行；改 metrics 列表或 metric_args 不必重跑 infer。

启动：
  python -m src.eval.run_metrics
  python -m src.eval.run_metrics --config configs/eval_base.yaml
  python -m src.eval.run_metrics --task answer
  python -m src.eval.run_metrics --task cot
  python -m src.eval.run_metrics --task all
"""

import argparse
from pathlib import Path
from typing import Any, Callable

import src.eval.metrics  # noqa: F401 — 触发 @register_*，否则 registry 为空
from src.common.config import load_config
from src.common.io import read_jsonl, write_json
from src.common.registry import get_answer_metric, get_cot_metric


def _join_pred_gold(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    按 join_key 对齐 pred 文件与 gold 文件，构造统一外壳。

    参数 cfg 为评测 yaml 中的 answer_metrics 或 cot_metrics 块，需包含：
      pred_file, gold_file, pred_field, gold_field, join_key（可选，默认 id）

    返回 list，每条为：
      {
        "id": ...,
        "pred": pred_row[pred_field] 或 "",
        "gold": gold_row[gold_field] 或 "",
        "pred_row": pred 文件中的原始一行 dict,
      }

    answer 与 cot 共用此外壳；字段语义差异在 pred_row 内部
    （如 pred_answer+strategy vs pred_cot+problem）。

    规则：
      - pred 文件决定样本顺序
      - pred 有而 gold 无 → 抛 KeyError
      - 文件不存在 → FileNotFoundError，并提示先跑 split
    """
    pred_path = Path(cfg["pred_file"])
    gold_path = Path(cfg["gold_file"])
    join_key = cfg.get("join_key", "id")
    pred_field = cfg["pred_field"]
    gold_field = cfg["gold_field"]

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}. "
            "Run `python -m src.eval.split` first."
        )
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}.")

    gold_by_key: dict[Any, dict[str, Any]] = {}
    for row in read_jsonl(gold_path):
        key = row.get(join_key)
        if key is None:
            raise KeyError(f"Gold row missing join_key {join_key!r}: {row}")
        if key in gold_by_key:
            raise ValueError(f"Duplicate gold {join_key}={key!r}")
        gold_by_key[key] = row

    joined: list[dict[str, Any]] = []
    for pred_row in read_jsonl(pred_path):
        key = pred_row.get(join_key)
        if key is None:
            raise KeyError(f"Pred row missing join_key {join_key!r}: {pred_row}")

        gold_row = gold_by_key.get(key)
        if gold_row is None:
            raise KeyError(
                f"No gold row for {join_key}={key!r}. "
                "Check pred_file vs gold_file alignment."
            )

        joined.append(
            {
                "id": key,
                "pred": pred_row.get(pred_field) or "",
                "gold": gold_row.get(gold_field) or "",
                "pred_row": pred_row,
            }
        )

    return joined


def _build_metric(
    name: str,
    metric_args: dict[str, Any] | None,
    *,
    cot: bool,
) -> Any:
    """
    从 registry 查找指标类并实例化。

    参数：
      name        — 配置 metrics 列表中的名字，如 exact_match / bert_score
      metric_args — 配置中 metric_args 整块；取 metric_args[name] 作为构造参数
      cot         — False 用 ANSWER_METRIC_REGISTRY，True 用 COT_METRIC_REGISTRY

    示例：acc 且 metric_args.acc.numeric=true → Accuracy(numeric=True)
    """
    getter: Callable[[str], type] = get_cot_metric if cot else get_answer_metric
    cls = getter(name)
    kwargs = (metric_args or {}).get(name, {})
    return cls(**kwargs)


def _run_one_metric(
    metric: Any,
    joined_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    cot: bool,
) -> float:
    """
    对单条指标名，在已对齐的 joined_rows 上计算聚合分数（通常 mean）。

    按 metrics.py 中指标类的接口分三种分支：
      1. apply_from_rows(pred_rows)     — extract_rate，读 pred_row.extract_ok
      2. apply_with_context(pred_rows)  — informativeness_chain，读 pred_cot + problem
      3. apply(preds, refs)             — exact_match / acc / bert_score

    参数 cot 当前仅作语义标记；分支由 metric 实例上实际存在的方法决定。

    返回：metric.aggregate(逐样本分数列表)
    """
    pred_rows = [row["pred_row"] for row in joined_rows]

    if hasattr(metric, "apply_from_rows"):
        scores = metric.apply_from_rows(pred_rows)
    elif hasattr(metric, "apply_with_context"):
        scores = metric.apply_with_context(
            pred_rows,
            pred_field=cfg["pred_field"],
            context_field=cfg.get("context_field", "problem"),
        )
    else:
        preds = [row["pred"] for row in joined_rows]
        refs = [row["gold"] for row in joined_rows]
        scores = metric.apply(preds, refs)

    return metric.aggregate(scores)


def _run_metrics_block(
    cfg: dict[str, Any],
    *,
    task_name: str,
    cot: bool,
) -> dict[str, Any]:
    """
    跑完 answer_metrics 或 cot_metrics 一整块配置。

    步骤：
      1. _join_pred_gold(cfg)
      2. 遍历 cfg["metrics"]，逐个 _build_metric + _run_one_metric
      3. 组装 result dict，write_json 到 cfg["output_file"]

    参数：
      task_name — 写入 JSON 的 "task" 字段，如 answer_metrics / cot_metrics
      cot       — 传给 _build_metric，决定用哪张注册表

    返回 result（含 metrics 分数；额外带 output_file 路径字符串）
    """
    joined_rows = _join_pred_gold(cfg)
    metric_args = cfg.get("metric_args") or {}
    metric_scores: dict[str, float] = {}

    for name in cfg.get("metrics", []):
        metric = _build_metric(name, metric_args, cot=cot)
        metric_scores[name] = _run_one_metric(metric, joined_rows, cfg, cot=cot)

    result: dict[str, Any] = {
        "task": task_name,
        "n_samples": len(joined_rows),
        "pred_file": cfg["pred_file"],
        "gold_file": cfg["gold_file"],
        "join_key": cfg.get("join_key", "id"),
        "metrics": metric_scores,
    }

    output_path = Path(cfg["output_file"])
    write_json(output_path, result)
    result["output_file"] = str(output_path)
    return result


def run_answer_metrics(config: dict[str, Any]) -> dict[str, Any]:
    """
    执行配置中的 answer_metrics 块。

    典型指标：exact_match, acc, extract_rate
    典型输出：reports/metrics/ft_answer.json
    """
    return _run_metrics_block(
        config["answer_metrics"],
        task_name="answer_metrics",
        cot=False,
    )


def run_cot_metrics(config: dict[str, Any]) -> dict[str, Any]:
    """
    执行配置中的 cot_metrics 块。

    典型指标：bert_score, informativeness_chain
    典型输出：reports/metrics/ft_cot.json
    """
    return _run_metrics_block(
        config["cot_metrics"],
        task_name="cot_metrics",
        cot=True,
    )


def run_metrics_pipeline(config: dict[str, Any], task: str = "all") -> dict[str, Any]:
    """
    主流程编排：按 task 决定跑 answer / cot / 两者。

    参数 task：
      "answer" — 只跑 answer_metrics
      "cot"    — 只跑 cot_metrics
      "all"    — 两者都跑（默认）

    返回 {"task": ..., "answer": {...}, "cot": {...}}，键随 task 裁剪。
    """
    if task not in {"answer", "cot", "all"}:
        raise ValueError(f"Unsupported task: {task}. Choose from: answer, cot, all.")

    results: dict[str, Any] = {"task": task}
    if task in {"answer", "all"}:
        results["answer"] = run_answer_metrics(config)
    if task in {"cot", "all"}:
        results["cot"] = run_cot_metrics(config)
    return results


def run_from_config(config_path: str | Path, task: str = "all") -> dict[str, Any]:
    """从 yaml 路径加载 config，再调用 run_metrics_pipeline。"""
    config = load_config(config_path)
    return run_metrics_pipeline(config, task=task)




def main() -> None:
    """CLI 入口：python -m src.eval.run_metrics"""
    parser = argparse.ArgumentParser(description="Run answer / cot evaluation metrics")
    parser.add_argument("--config", default="configs/eval_ft.yaml", help="评测配置文件路径")
    parser.add_argument(
        "--task",
        default="all",
        choices=["answer", "cot", "all"],
        help="只跑 answer、cot，或两者都跑",
    )
    args = parser.parse_args()

    results = run_from_config(args.config, task=args.task)
    print("run_metrics done:")
    print(f"  task: {results['task']}")
    if "answer" in results:
        ans = results["answer"]
        print(f"  answer output: {ans['output_file']} (n={ans['n_samples']})")
        for name, value in ans["metrics"].items():
            print(f"    {name}: {value:.4f}")
    if "cot" in results:
        cot = results["cot"]
        print(f"  cot output: {cot['output_file']} (n={cot['n_samples']})")
        for name, value in cot["metrics"].items():
            print(f"    {name}: {value:.4f}")


if __name__ == "__main__":
    main()
