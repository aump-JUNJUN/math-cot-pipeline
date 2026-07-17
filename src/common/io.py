"""
读写项目中的文件
1. 读取JSON文件 train.jsonl  test.jsonl 及report的文件  read_jsonl(path)
2. 写入JSON文件 train.jsonl  test.jsonl 及report的文件  write_jsonl(path,row)  write_jsonl(path,data)
3. try except 捕获异常，并给出错误信息
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """逐行读取 jsonl，每行 yield 一个 dict。"""
    file_path = Path(path)
    with file_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip() #去除额外字符

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {file_path}"
                ) from exc



def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """把多条记录写入 jsonl，返回写入行数。"""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with file_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count



def write_json(path: str | Path, data: Any) -> None:
    """写入普通 json 文件（如 manifest.json）。"""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    from pathlib import Path

    tmp = Path("data/processed/_io_test.jsonl")
    n = write_jsonl(tmp, [{"id": "1", "problem": "1+1"}, {"id": "2", "problem": "2+2"}])
    print("written:", n)

    for row in read_jsonl(tmp):
        print(row)

    write_json("data/processed/_io_test.json", {"ok": True, "count": n})
    tmp.unlink(missing_ok=True)


