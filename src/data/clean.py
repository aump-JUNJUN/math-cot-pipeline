"""
输入：
1. config中的配置  2. 数据集 data/raw/numina_math_cot.jsonl
输出：
跟我的schema数据格式相符的cleaned_data中间结果数据集

大致过程：
1. 读配置YAML：common  读JSON: common
2. 清洗数据
总目标：1. 清洗空行，其中包括question answer的空缺  2. 控制长度：question answer  3. 去重  4. 抽取COT 和 answer → common 
        6. 数据映射 → config中schema的映射字段

伪代码：
def clean_data(config: str | Path) -> dict[str, Any]:
    1. 读取配置文件：config = load_config(config_path)
    2. 读取数据集：dataset/raw/numina_math_cot.jsonl

    cleaned_rows = []
    3. for row  in  read_jsonl(dataset/raw/numina_math_cot.jsonl):
        def clean_row(row,cleaning) return cleaned_row
            if cleaned 存在:
                cleaned_rows.append(cleaned_row)


    4. 去重 
    dedupe_rows = dedupe(cleaned_rows)

    再次按照row来遍历
    抽取COT 和 answer 
    for row in dedupe_rows:
        def extract_cot(row) return cot,answer
            if cot 存在:
                dedupe_rows.append(cot)
            if answer 存在:
                dedupe_rows.append(answer)


    write_jsonl(dedupe_rows,schema_path)
   
"""

from __future__ import annotations

"""
读取 data/raw/*.jsonl，清洗 → 去重 → 划分 train/test → 写入 data/processed/。

清洗步骤（在 _clean_row 内一次完成）：
  空行 / 长度 / split_solution 抽 COT+answer / drop_no_answer
"""

import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common.config import load_config
from src.common.extract import ExtractResult, split_solution
from src.common.io import read_jsonl, write_json, write_jsonl


def _clean_row(
    row: dict[str, Any],  #一行变为一个dict
    cleaning: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """单行清洗：去空、控长度、抽 COT/answer，失败则返回丢弃原因。"""
    problem = (row.get("problem") or "").strip()
    solution = (row.get("solution") or "").strip()

    if cleaning.get("drop_empty", True):
        if not problem:
            return None, "empty_problem"
        if not solution:
            return None, "empty_solution"

    max_q = cleaning.get("max_question_chars")
    if max_q is not None and len(problem) > int(max_q):
        return None, "too_long_question"

    max_s = cleaning.get("max_solution_chars")
    if max_s is not None and len(solution) > int(max_s):
        return None, "too_long_solution"

    extract_cfg = cleaning.get("answer_extract") or {}
    if extract_cfg.get("enabled", True):
        strategies = extract_cfg.get("strategies", ["boxed"])
        result = split_solution(solution, strategies)
    else:
        result = ExtractResult(answer=None, cot=solution, extract_ok=False)

    if cleaning.get("drop_no_answer", True) and not result.extract_ok:
        return None, "no_answer"

    cleaned: dict[str, Any] = {
        "source": row.get("source"),
        "problem": problem,
        "solution": solution,
        "COT": result.cot,
        "answer": result.answer,
    }
    if "message" in row:
        cleaned["message"] = row["message"]

    return cleaned, None


def _question_hash(problem: str) -> str:
    """对题干做 sha256，用作去重 key。"""
    return hashlib.sha256(problem.strip().encode("utf-8")).hexdigest()


def _dedupe_rows(
    rows: list[dict[str, Any]],
    dedupe_by: str | None,
) -> list[dict[str, Any]]:
    """按 config 指定字段去重，保留首次出现的样本。"""
    if not dedupe_by:
        return rows

    if dedupe_by != "question_hash":
        raise ValueError(f"Unsupported dedupe_by: {dedupe_by}")

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = _question_hash(row["problem"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _assign_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为每条样本分配固定格式的 id（000000, 000001, ...）。"""
    assigned: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        new_row = dict(row)
        new_row["id"] = f"{idx:06d}"
        assigned.append(new_row)
    return assigned


def _split_rows(
    rows: list[dict[str, Any]],
    split_cfg: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按 seed 打乱后，按 test_ratio 切成 train 和 test。"""
    rows_copy = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows_copy)

    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    test_size = int(len(rows_copy) * test_ratio)

    if split_cfg.get("holdout_test", True) and test_size == 0 and len(rows_copy) > 1:
        test_size = 1

    test_rows = rows_copy[:test_size]
    train_rows = rows_copy[test_size:]
    return train_rows, test_rows


def _project_schema(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:

    """只保留 schema 里声明的字段，写入 jsonl 前做列投影。"""
    # field: row.get(field) 的意思是：以 field 为键，row.get(field) 为值，组成一个新的字典项
    # 也就是用 fields 里的每个字段名，在 row 这条字典里查询对应的值
    # 比如 fields 里有 "source"，那 row.get("source") 就是获取 row 里 key 为 "source" 的值

    projected = {}
    for field in fields:
        projected[field] = row.get(field)
    return projected


def run_clean_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """主流程：读 raw → 清洗 → 去重 → 划分 → 写 train/test 和 manifest。"""
    cleaning = config["cleaning"]
    split_cfg = config.get("split", {})
    output_cfg = config["output"]
    schema_fields = config.get("schema", {}).get("fields", [])
    seed = int(config.get("seed", 42))
    raw_dir = Path(config["raw_dir"])
    processed_dir = Path(config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    cleaned_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "raw": 0,
        "kept_after_clean": 0,
        "after_dedupe": 0,
        "train": 0,
        "test": 0,
        "dropped": defaultdict(int),
    }

    for source_name, source_cfg in config.get("sources", {}).items():
        if not source_cfg.get("enabled", True):
            continue

        raw_path = raw_dir / f"{source_name}.jsonl"

        for row in read_jsonl(raw_path):
            stats["raw"] += 1
            cleaned, reason = _clean_row(row, cleaning)
            if cleaned is None:
                stats["dropped"][reason or "unknown"] += 1
                continue
            cleaned_rows.append(cleaned)

    stats["kept_after_clean"] = len(cleaned_rows)

    dedupe_by = cleaning.get("dedupe_by")
    cleaned_rows = _dedupe_rows(cleaned_rows, dedupe_by)
    stats["after_dedupe"] = len(cleaned_rows)

    cleaned_rows = _assign_ids(cleaned_rows)
    train_rows, test_rows = _split_rows(cleaned_rows, split_cfg, seed)


    train_out = [_project_schema(r, schema_fields) for r in train_rows]
    test_out = [_project_schema(r, schema_fields) for r in test_rows]


    train_path = Path(output_cfg["train_file"])
    test_path = Path(output_cfg["test_file"])
    manifest_path = Path(output_cfg["manifest_file"])

    train_written = write_jsonl(train_path, train_out)
    test_written = write_jsonl(test_path, test_out)

    stats["train"] = train_written
    stats["test"] = test_written
    stats["dropped"] = dict(stats["dropped"])

    manifest: dict[str, Any] = {
        "processed_dir": str(processed_dir),
        "raw_dir": str(raw_dir),
        "seed": seed,
        "cleaning": cleaning,
        "split": split_cfg,
        "schema_fields": schema_fields,
        "stats": stats,
        "output": {
            "train_file": str(train_path),
            "test_file": str(test_path),
            "train_rows": train_written,
            "test_rows": test_written,
        },
    }

    write_json(manifest_path, manifest)
    manifest["manifest_file"] = str(manifest_path)
    return manifest


def run_from_config(config_path: str | Path) -> dict[str, Any]:
    """从 yaml 路径加载 config，再调用 run_clean_pipeline。"""
    config = load_config(config_path)
    return run_clean_pipeline(config)


if __name__ == "__main__":
    result = run_from_config("configs/data.yaml")
    stats = result["stats"]
    print(
        f"raw={stats['raw']} -> kept={stats['kept_after_clean']} "
        f"-> deduped={stats['after_dedupe']}"
    )
    print(f"train={stats['train']} test={stats['test']}")
    print("dropped:", stats["dropped"])
    print("manifest:", result["manifest_file"])

    