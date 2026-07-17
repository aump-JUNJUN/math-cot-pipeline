from __future__ import annotations
#下载数据集，从 Hugging Face 下载 NuminaMath-CoT，存到 data/raw/   JSON格式
"""
既要数据文件，也要一份「下载收据」
1. 从 Hugging Face 下载 NuminaMath-CoT
with语法
source ： olympiads
数量1000条
产出的字段对应yaml文件中的config
"""

"""
从 Hugging Face 下载 NuminaMath-CoT，按 config 过滤（如 source=olympiads），写入 data/raw/*.jsonl。
"""


from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

from src.common.config import load_config
from src.common.io import write_json, write_jsonl


def _map_record(row: dict[str, Any], field_map: dict[str, str]) -> dict[str, Any]:
    """
    按 field_map 把 HuggingFace 字段映射成 raw jsonl 字段。
    返回的record格式如下（以 field_map = {"problem": "problem", "solution": "solution", "source": "source"} 为例）：

    {
        "problem": ...,   # 对应 row["problem"]
        "solution": ...,  # 对应 row["solution"]
        "source": ...,    # 对应 row["source"]
        ...
    }

    其中 key 为目标字段名（最终保存在 jsonl 里的列名），value 为原始 row 里相应字段的值。
    """
    record: dict[str, Any] = {}
    for target, source_field in field_map.items():
        record[target] = row.get(source_field)
    return record


def _apply_filter(dataset, filter_cfg: dict[str, Any] | None):
  
    if not filter_cfg:
        return dataset

    field = filter_cfg["field"]
    values = set(filter_cfg.get("values", []))
    if not values:
        return dataset

    # 这里的 filter 是 HuggingFace datasets 库中 Dataset 对象自带的方法，用于对数据集进行筛选（过滤）操作，
    # 并不是外部传入的 filter_cfg，也不是要调用哪个外部过滤函数。
    # 具体用法就是用一个 lambda 进行条件判定，保留满足条件的行（样本）。
    return dataset.filter(lambda row: row.get(field) in values)


def download_source(
    source_name: str,
    source_cfg: dict[str, Any],
    raw_dir: str | Path,
) -> dict[str, Any]:

    hf_id = source_cfg["hf_id"]
    split_name = source_cfg.get("split_download", "train")
    max_samples = source_cfg.get("max_samples")
    field_map = source_cfg.get("field_map", {})
    filter_cfg = source_cfg.get("filter")

    dataset = load_dataset(hf_id, split=split_name)
    total_before_filter = len(dataset)

    dataset = _apply_filter(dataset, filter_cfg)
    total_after_filter = len(dataset)

    if max_samples is not None:
        max_samples = int(max_samples)
        # 是的，这里的 select 也是 HuggingFace datasets 库提供的 Dataset 对象的方法，用于按索引选择子集。
        dataset = dataset.select(range(min(max_samples, len(dataset))))
 

    # 这是将处理/筛选后的数据集写回本地 raw_dir，保存为 jsonl，每行为一个样本。
    # - raw_path: 保存的文件路径，比如 data/raw/numina_math_cot.jsonl
    # - rows: 经过字段映射的记录生成器，这里按 field_map 取出目标字段
    # - tqdm: 进度条显示写入进度
    # - write_jsonl: 实际把 rows 写入 raw_path 文件，并返回写入的条数
    raw_path = Path(raw_dir) / f"{source_name}.jsonl"
    rows = (
        _map_record(row, field_map)
        for row in tqdm(dataset, desc=f"Writing {source_name}")
    )
    written = write_jsonl(raw_path, rows)
    # 这里 return 的是一个包含各种数据下载和处理信息的字典（manifest 结构的一部分），主要字段有：
    # - source_name: 数据源名字（如 numina_math_cot）
    # - hf_id: HuggingFace 数据集 ID
    # - split: 下载的数据集 split 名（如 train）
    # - total_before_filter: 筛选前总样本数
    # - total_after_filter: 筛选后剩余样本数
    # - written: 实际写入本地 raw 文件的样本条数
    # - raw_file: 写入的本地文件路径
    # - filter: 采用的过滤配置
    return {
        "source_name": source_name,
        "hf_id": hf_id,
        "split": split_name,
        "total_before_filter": total_before_filter,
        "total_after_filter": total_after_filter,
        "written": written,
        "raw_file": str(raw_path),
        "filter": filter_cfg,
    }



def download_sources(config: dict[str, Any]) -> dict[str, Any]:
    """
    这个函数的主要工作就是根据 config 的 sources 配置，批量下载/筛选所有启用的数据源，把每个源的数据写到 raw_dir 目录下，
    然后组织所有源的关键信息（比如写入了多少条/文件名等）打一个“总 manifest”出来 download_manifest.json，用于记录和追踪下载的全过程。

    也就是说，这一步只负责【下载 → 写原始文件 → 生成 manifest 记录】，不会直接进行后续的数据清洗/切分等步骤。
    后续比如 clean/split 操作都是在 raw_dir 产物和 manifest 的基础上再做的。

    返回结果 manifest 里，把所有源每次下载和筛选的详细情况都清楚地罗列了一遍，并记录 manifest_file 路径，
    可以作为后续数据流转和溯源的唯一入口记录。
    """
    raw_dir = Path(config["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "raw_dir": str(raw_dir),
        "sources": [],
    }


    for source_name, source_cfg in config.get("sources", {}).items():
        if not source_cfg.get("enabled", True):
            continue
        manifest["sources"].append(
            download_source(source_name, source_cfg, raw_dir)
        )

    manifest_path = raw_dir / "download_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_file"] = str(manifest_path)
    return manifest


def run_from_config(config_path: str | Path) -> dict[str, Any]:
    # config 就是配置文件（如 configs/data.yaml）加载后的 Python 字典
    config = load_config(config_path)
    return download_sources(config)



if __name__ == "__main__":
    result = run_from_config("configs/data.yaml")
    for item in result["sources"]:
        print(
            f"{item['source_name']}: "
            f"filter {item['total_before_filter']} -> {item['total_after_filter']}, "
            f"written {item['written']} -> {item['raw_file']}"
        )
    print("manifest:", result["manifest_file"])


