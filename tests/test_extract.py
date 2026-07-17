"""
单元测试：src/common/extract.py

背景
----
NuminaMath-CoT 的 solution / 模型 generated_text 通常把最终答案写在 \\boxed{...} 里。
extract 模块负责把整段文本拆成：
  - answer：boxed 内的最终答案（用于 EM / acc / extract_rate）
  - cot：boxed 之前的推理过程（用于 CoT 指标或训练金标）

该模块被两条链路共用：
  1. src/data/clean.py   — 清洗 raw 数据，写 train.jsonl / test.jsonl 的金标
  2. src/eval/split.py   — 拆分 predictions/*.jsonl 的 pred_answer / pred_cot

config 里 strategies 目前只有 ["boxed"]（见 configs/data.yaml、eval_*.yaml）。

本文件测什么
------------
| 函数 / 场景              | 为何重要                                      |
|-------------------------|-----------------------------------------------|
| split_solution 正常 boxed | 主路径；COT 与 answer 分界正确                  |
| 多个 \\boxed             | 模型可能重复写答案；约定取最后一个              |
| \\fbox{...}              | 与 \\boxed 等价的备选 marker                   |
| 嵌套花括号               | 数学答案常见 \\frac{1}{2} 等 LaTeX             |
| 无 / 空 / 未闭合 boxed   | extract_ok=False → clean 剔除、extract_rate  |
| extract_answer strategy | 默认 boxed；未知 strategy 应明确报错            |

运行
----
  pip install -r requirements/dev.txt
  pytest tests/test_extract.py -v
"""

from __future__ import annotations

import pytest

from src.common.extract import extract_answer, extract_boxed, split_solution


# ── split_solution：主入口（clean + eval split 都用它）────────────────────────


def test_split_solution_basic_boxed():
    """标准 CoT + \\boxed{answer}：extract_ok=True，COT 不含 boxed 标记。"""
    text = (
        "Step 1: compute 2+2=4.\n"
        "Step 2: multiply by 3.\n"
        "Therefore the answer is \\boxed{12}"
    )
    result = split_solution(text, ["boxed"])

    assert result.extract_ok is True
    assert result.answer == "12"
    assert result.strategy == "boxed"
    assert "Step 1" in result.cot
    assert "\\boxed" not in result.cot


def test_split_solution_no_boxed():
    """无 boxed → extract_ok=False；answer=None，cot 保留全文（strip 后）。"""
    text = "I think the answer is 12 but no marker."
    result = split_solution(text, ["boxed"])

    assert result.extract_ok is False
    assert result.answer is None
    assert result.cot == text
    assert result.strategy is None


def test_split_solution_empty_boxed():
    """\\boxed{} 内容为空视为抽不到（extract_answer 会跳过空 value）。"""
    text = "Reasoning here. \\boxed{}"
    result = split_solution(text, ["boxed"])

    assert result.extract_ok is False
    assert result.answer is None


def test_split_solution_unclosed_boxed():
    """未闭合的花括号无法解析 → 视为抽不到。"""
    text = "Broken \\boxed{12"
    result = split_solution(text, ["boxed"])

    assert result.extract_ok is False
    assert result.answer is None


# ── extract_boxed：底层 marker 扫描 ──────────────────────────────────────────


def test_extract_boxed_takes_last():
    """多个 \\boxed 时取 start 位置最靠后的（最后一个）。"""
    text = "First \\boxed{wrong} then final \\boxed{42}"
    span = extract_boxed(text)

    assert span is not None
    assert span.value == "42"
    assert span.strategy == "boxed"


def test_extract_fbox():
    """\\fbox{...} 与 \\boxed{...} 同等支持。"""
    text = "Answer: \\fbox{7}"
    span = extract_boxed(text)

    assert span is not None
    assert span.value == "7"


def test_nested_braces_in_boxed():
    """LaTeX 嵌套花括号：\\frac{1}{2} 应完整落入 answer。"""
    text = "So \\boxed{\\frac{1}{2}}"
    result = split_solution(text, ["boxed"])

    assert result.extract_ok is True
    assert result.answer == "\\frac{1}{2}"


def test_boxed_value_is_stripped():
    """boxed 内首尾空白应被 strip。"""
    text = "Thus \\boxed{  99  }"
    result = split_solution(text, ["boxed"])

    assert result.extract_ok is True
    assert result.answer == "99"


# ── extract_answer：strategy 路由（config strategies 入口）────────────────────


def test_extract_answer_default_strategy():
    """strategies=None 时默认走 boxed。"""
    span = extract_answer("\\boxed{99}")

    assert span is not None
    assert span.value == "99"


def test_extract_answer_unsupported_strategy():
    """尚未实现的 strategy 应抛 ValueError，避免静默失败。"""
    with pytest.raises(ValueError, match="Unsupported strategy"):
        extract_answer("\\boxed{1}", strategies=["regex"])
