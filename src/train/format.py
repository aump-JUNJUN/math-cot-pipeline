from __future__ import annotations

"""
读取 train.jsonl，按 train_schema 转成 ms-swift 所需的 messages jsonl。

输入：configs/train.yaml
  - data.train_file
  - data.train_schema
输出：
  - data.messages_file（如 data/processed/train_messages.jsonl）

启动：
  python -m src.train.format
  python -m src.train.format --config configs/train.yaml
"""

import argparse
import copy
import re
from pathlib import Path
from typing import Any

from src.common.config import load_config
from src.common.io import read_jsonl, write_jsonl


# 这是一个用于匹配形如 <problem> 这样占位符（尖括号包裹的单词）的正则表达式
_PLACEHOLDER_RE = re.compile(r"<(\w+)>")


def _fill_placeholders(text: str, row: dict[str, Any]) -> str:
    """把 "<problem>" 等占位符替换成 row 里对应字段的值。"""
    def replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        value = row.get(key)
        if value is None:
            return match.group(0)
        return str(value)

    return _PLACEHOLDER_RE.sub(replacer, text)


def _apply_schema(row: dict[str, Any], train_schema: dict[str, Any]) -> dict[str, Any]:
    """单行样本按 train_schema 转成 {"messages": [...]}。"""
    schema_copy = copy.deepcopy(train_schema)
    messages = schema_copy.get("messages", [])

    formatted_messages: list[dict[str, str]] = []
    for message in messages:
        formatted_messages.append(
            {
                "role": message["role"],
                "content": _fill_placeholders(message["content"], row),
            }
        )

    return {"messages": formatted_messages}


def build_messages_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """主流程：读 train.jsonl → 转 messages → 写 messages_file。"""
    data_cfg = config["data"]
    train_path = Path(data_cfg["train_file"])
    messages_path = Path(data_cfg["messages_file"])
    train_schema = data_cfg["train_schema"]

    rows: list[dict[str, Any]] = []
    for row in read_jsonl(train_path):
        rows.append(_apply_schema(row, train_schema))

    written = write_jsonl(messages_path, rows)

    return {
        "train_file": str(train_path),
        "messages_file": str(messages_path),
        "rows": written,
    }


def run_from_config(config_path: str | Path) -> dict[str, Any]:
    """从 yaml 路径加载 config，再调用 build_messages_dataset。"""
    config = load_config(config_path)
    return build_messages_dataset(config)


"""   CLI 入口  """

def main() -> None:
    """CLI 入口：python -m src.train.format"""
    parser = argparse.ArgumentParser(description="Convert train.jsonl to train_messages.jsonl")
    parser.add_argument("--config", default="configs/train.yaml", help="训练配置文件路径")
    args = parser.parse_args()

    result = run_from_config(args.config)
    print("format done:")
    print(f"  train_file: {result['train_file']}")
    print(f"  messages_file: {result['messages_file']}")
    print(f"  rows: {result['rows']}")


if __name__ == "__main__":
    main()
