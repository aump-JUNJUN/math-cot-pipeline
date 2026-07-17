"""
1. 文件中输入的是配置路径 configs/data.yaml
2. 文件的输出是python dict
"""
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


if __name__ == "__main__":
    cfg = load_config("configs/data.yaml")
    print(cfg["seed"])           # 42
    print(cfg["cleaning"].keys())