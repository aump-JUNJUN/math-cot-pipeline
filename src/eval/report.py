"""
评测流水线第 4 步：对比多条 eval 链的聚合指标（CPU，无需 GPU）。

读 configs/compare.yaml 中 compare.runs 列出的 metrics JSON，
写 compare.output_file（默认 reports/metrics/compare_all.json），
可选写 compare.plot_file（默认 reports/metrics/compare_all.png）。

不重新算指标；不读 pred jsonl。

前置：各 run 已跑完 infer → split → run_metrics
  base → configs/eval_base.yaml
  ft   → configs/eval_ft.yaml

启动：
  python -m src.eval.report
  python -m src.eval.report --no-plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common.config import load_config
from src.common.io import write_json


def _load_metrics(path: str | Path) -> dict[str, Any]:
    """读取 run_metrics 产出的 JSON；必须含 metrics 字段。"""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {file_path}. "
            "Run `python -m src.eval.run_metrics` for each run first."
        )

    with file_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if "metrics" not in data or not isinstance(data["metrics"], dict):
        raise ValueError(f"Invalid metrics JSON (missing 'metrics'): {file_path}")

    return data


def _parse_compare_config(config: dict[str, Any]) -> dict[str, Any]:
    """解析 compare.yaml 的 compare 块。"""
    if "compare" not in config:
        raise ValueError(
            "Config must contain a 'compare' block. Use configs/compare.yaml."
        )

    cmp_cfg = config["compare"]
    output_file = Path(
        cmp_cfg.get("output_file", "reports/metrics/compare_all.json")
    )
    baseline = cmp_cfg.get("baseline")
    runs = cmp_cfg.get("runs")

    if not runs or not isinstance(runs, list):
        raise ValueError("compare.runs must be a non-empty list.")

    parsed_runs: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"compare.runs[{idx}] must be a mapping.")
        name = run.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"compare.runs[{idx}] missing valid 'name'.")
        if name in seen_names:
            raise ValueError(f"Duplicate run name in compare.runs: {name!r}")
        seen_names.add(name)

        answer_path = run.get("answer_metrics")
        cot_path = run.get("cot_metrics")
        if not answer_path or not cot_path:
            raise ValueError(
                f"compare.runs[{idx}] ({name!r}) must set "
                "'answer_metrics' and 'cot_metrics'."
            )
        parsed_runs.append(
            {
                "name": name,
                "answer_metrics": Path(answer_path),
                "cot_metrics": Path(cot_path),
            }
        )

    if baseline is not None:
        if not isinstance(baseline, str):
            raise ValueError("compare.baseline must be a run name or null.")
        if baseline not in seen_names:
            raise ValueError(
                f"compare.baseline={baseline!r} not found in compare.runs."
            )

    if "plot_file" not in cmp_cfg:
        plot_file: Path | None = output_file.with_suffix(".png")
    elif cmp_cfg["plot_file"] is None:
        plot_file = None
    else:
        plot_file = Path(cmp_cfg["plot_file"])

    return {
        "output_file": output_file,
        "plot_file": plot_file,
        "baseline": baseline,
        "runs": parsed_runs,
    }


def _load_all_runs(
    runs: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    """加载每个 run 的 answer / cot metrics 文档。"""
    docs: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, str]] = {}

    for run in runs:
        name = run["name"]
        answer_path = run["answer_metrics"]
        cot_path = run["cot_metrics"]
        docs[name] = {
            "answer": _load_metrics(answer_path),
            "cot": _load_metrics(cot_path),
        }
        sources[name] = {
            "answer_metrics": str(answer_path),
            "cot_metrics": str(cot_path),
        }

    return docs, sources


def _compare_side(
    run_docs: dict[str, dict[str, Any]],
    *,
    side: str,
    baseline: str | None,
) -> dict[str, dict[str, Any]]:
    """
    对比 answer 或 cot 一侧的所有指标。

    只对比所有 run 都存在的指标名。
    baseline 非 null 时，delta[run] = values[run] - values[baseline]（不含 baseline 自身）。
    """
    side_docs = {name: data[side] for name, data in run_docs.items()}

    sample_counts = {
        name: doc.get("n_samples")
        for name, doc in side_docs.items()
        if doc.get("n_samples") is not None
    }
    if len(set(sample_counts.values())) > 1:
        details = ", ".join(f"{name}={count}" for name, count in sample_counts.items())
        raise ValueError(f"n_samples mismatch for {side}: {details}")

    metric_sets = [set(doc["metrics"].keys()) for doc in side_docs.values()]
    common = set.intersection(*metric_sets) if metric_sets else set()
    if not common:
        raise ValueError(
            f"No overlapping metrics for {side}. "
            "Check metrics JSON files and metric names."
        )

    out: dict[str, dict[str, Any]] = {}
    for metric_name in sorted(common):
        values = {
            run: float(side_docs[run]["metrics"][metric_name])
            for run in side_docs
        }
        entry: dict[str, Any] = {"values": values}
        if baseline is not None:
            base_val = values[baseline]
            entry["delta"] = {
                run: val - base_val
                for run, val in values.items()
                if run != baseline
            }
        out[metric_name] = entry

    return out


def _save_compare_plot(
    result: dict[str, Any],
    plot_path: Path,
    *,
    baseline: str | None,
) -> None:
    """将 answer / cot 对比结果画成分组柱状图并保存为 PNG。"""
    import matplotlib.pyplot as plt
    import numpy as np

    sides = [("Answer metrics", result["answer"]), ("CoT metrics", result["cot"])]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    for ax, (title, side_data) in zip(axes, sides):
        if not side_data:
            ax.set_visible(False)
            continue

        metric_names = list(side_data.keys())
        run_names = list(next(iter(side_data.values()))["values"].keys())
        n_metrics = len(metric_names)
        n_runs = len(run_names)
        x = np.arange(n_metrics)
        width = 0.8 / max(n_runs, 1)
        max_val = 0.0

        for i, run in enumerate(run_names):
            vals = [side_data[m]["values"][run] for m in metric_names]
            max_val = max(max_val, max(vals))
            offset = (i - (n_runs - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=run)
            if run == baseline:
                for bar in bars:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(1.5)
            for j, bar in enumerate(bars):
                delta = side_data[metric_names[j]].get("delta", {})
                if run in delta:
                    ax.annotate(
                        f"{delta[run]:+.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=15, ha="right")
        ax.set_ylim(0, max(max_val * 1.15, 0.1))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def run_compare_pipeline(config: dict[str, Any], *, plot: bool = True) -> dict[str, Any]:
    """主流程：加载各 run 的 answer + cot metrics → 写 compare JSON（可选 PNG）。"""
    cmp_cfg = _parse_compare_config(config)
    run_docs, sources = _load_all_runs(cmp_cfg["runs"])
    baseline = cmp_cfg["baseline"]

    result: dict[str, Any] = {
        "task": "compare",
        "baseline": baseline,
        "answer": _compare_side(run_docs, side="answer", baseline=baseline),
        "cot": _compare_side(run_docs, side="cot", baseline=baseline),
        "sources": sources,
    }

    output_file = cmp_cfg["output_file"]
    write_json(output_file, result)
    result["compare_file"] = str(output_file)

    plot_file = cmp_cfg["plot_file"]
    if plot and plot_file is not None:
        try:
            _save_compare_plot(result, plot_file, baseline=baseline)
            result["plot_file"] = str(plot_file)
        except ImportError:
            print("warning: matplotlib not installed, skipping plot")

    return result


def run_from_config(config_path: str | Path, *, plot: bool = True) -> dict[str, Any]:
    """从 yaml 路径加载 config，再调用 run_compare_pipeline。"""
    config = load_config(config_path)
    return run_compare_pipeline(config, plot=plot)


def _print_side(side_name: str, side_result: dict[str, dict[str, Any]]) -> None:
    print(f"  {side_name}:")
    for metric_name, entry in side_result.items():
        values_text = "  ".join(
            f"{run}={val:.4f}" for run, val in entry["values"].items()
        )
        line = f"    {metric_name}: {values_text}"
        delta = entry.get("delta")
        if delta:
            delta_text = "  ".join(f"Δ({run})={val:+.4f}" for run, val in delta.items())
            line = f"{line}  {delta_text}"
        print(line)


def main() -> None:
    """CLI 入口：python -m src.eval.report"""
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics across runs (configs/compare.yaml)"
    )
    parser.add_argument(
        "--config",
        default="configs/compare.yaml",
        help="对比配置文件路径（默认 configs/compare.yaml）",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="只写 JSON，不生成 PNG",
    )
    args = parser.parse_args()

    result = run_from_config(args.config, plot=not args.no_plot)
    print("report done:")
    print(f"  compare_file: {result['compare_file']}")
    if result.get("plot_file"):
        print(f"  plot_file:    {result['plot_file']}")
    if result["baseline"] is not None:
        print(f"  baseline: {result['baseline']}")
    _print_side("answer", result["answer"])
    _print_side("cot", result["cot"])


if __name__ == "__main__":
    main()
