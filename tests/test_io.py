"""
单元测试：src/common/io.py

背景
----
io 模块负责项目里 jsonl / json 的读写，是数据与评测链路的底层 I/O：
  1. src/data/clean.py      — 写 train.jsonl / test.jsonl / manifest.json
  2. src/eval/infer.py      — 读 test.jsonl，写 predictions/*.jsonl
  3. src/eval/split.py      — 读 predictions，写 *_answers.jsonl / *_cots.jsonl
  4. src/eval/run_metrics.py — 读上述 jsonl，写 metrics/*.json

若读写行为出错（空行、非法 JSON、目录不存在），会导致整条 pipeline 静默失败或报错不清晰。

本文件测什么
------------
| 函数 / 场景                    | 为何重要                           |
|-------------------------------|------------------------------------|
| write_jsonl + read_jsonl 往返 | 主路径；数据与 predictions 依赖它  |
| read_jsonl 跳过空行            | jsonl 文件常有尾部空行             |
| read_jsonl 非法 JSON           | 应抛带行号的 ValueError            |
| write_jsonl 自动建目录         | reports/ 等目录可能尚不存在        |
| write_jsonl 返回行数           | clean 等模块用返回值做统计         |
| write_json 写 manifest       | manifest.json 等元数据             |
| ensure_ascii=False           | 中文题目 / 解答不能变成 \\uXXXX    |

运行
----
  pip install -r requirements/dev.txt
  pytest tests/test_io.py -v
"""

from __future__ import annotations

import json

import pytest

from src.common.io import read_jsonl, write_json, write_jsonl


# ── jsonl 往返 ────────────────────────────────────────────────────────────────


def test_write_read_jsonl_roundtrip(tmp_path):
    """写入两行 jsonl 再读出，字段应完全一致。"""
    path = tmp_path / "sample.jsonl"
    rows = [
        {"id": "1", "problem": "1+1"},
        {"id": "2", "problem": "2+2", "answer": "4"},
    ]

    written = write_jsonl(path, rows)
    loaded = list(read_jsonl(path))

    assert written == 2
    assert loaded == rows


def test_read_jsonl_skips_blank_lines(tmp_path):
    """空行应被忽略，不影响有效行的解析。"""
    path = tmp_path / "with_blanks.jsonl"
    path.write_text(
        '{"id": "1"}\n\n  \n{"id": "2"}\n',
        encoding="utf-8",
    )

    loaded = list(read_jsonl(path))

    assert len(loaded) == 2
    assert loaded[0]["id"] == "1"
    assert loaded[1]["id"] == "2"


def test_read_jsonl_invalid_json_raises(tmp_path):
    """损坏的行应抛 ValueError，并指明行号与文件路径。"""
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid JSON on line 2"):
        list(read_jsonl(path))


def test_write_jsonl_creates_parent_dirs(tmp_path):
    """目标目录不存在时 write_jsonl 应自动创建。"""
    path = tmp_path / "nested" / "dir" / "out.jsonl"

    n = write_jsonl(path, [{"id": "x"}])

    assert n == 1
    assert path.is_file()


def test_write_jsonl_empty_iterable(tmp_path):
    """空输入应写出 0 行并创建空文件。"""
    path = tmp_path / "empty.jsonl"

    n = write_jsonl(path, [])

    assert n == 0
    assert path.read_text(encoding="utf-8") == ""


def test_write_jsonl_preserves_unicode(tmp_path):
    """ensure_ascii=False：中文等非 ASCII 字符原样保留。"""
    path = tmp_path / "unicode.jsonl"
    row = {"problem": "计算 1+1", "answer": "二"}

    write_jsonl(path, [row])
    raw = path.read_text(encoding="utf-8")

    assert "计算" in raw
    assert "\\u" not in raw
    assert list(read_jsonl(path)) == [row]


# ── write_json（manifest 等）──────────────────────────────────────────────────


def test_write_json_roundtrip(tmp_path):
    """write_json 写出可 json.load 的标准 JSON（带缩进）。"""
    path = tmp_path / "manifest.json"
    data = {"train": 778, "test": 86, "ok": True}

    write_json(path, data)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == data


def test_write_json_creates_parent_dirs(tmp_path):
    """write_json 同样应自动创建父目录。"""
    path = tmp_path / "meta" / "info.json"

    write_json(path, {"version": 1})

    assert path.is_file()
