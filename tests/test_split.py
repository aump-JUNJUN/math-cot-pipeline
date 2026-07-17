"""
单元测试：src/eval/split.py

背景
----
split 是评测流水线第 2 步（CPU，无需 GPU）：
  infer.py  → predictions/{name}.jsonl（含 generated_text）
  split.py  → {name}_answers.jsonl（pred_answer）
              {name}_cots.jsonl（pred_cot）
  run_metrics.py → 分别算答案侧 / CoT 侧指标

核心逻辑：
  1. 对 generated_text 调用 common.extract.split_solution（见 test_extract.py）
  2. _apply_cot_mode 决定 pred_cot 写「boxed 前推理」还是「整段原文」
  3. run_split_pipeline 批量读写 jsonl，并统计 extract_rate

configs/eval_ft.yaml → split.cot_mode 默认为 before_answer。

本文件测什么
------------
| 函数 / 场景                         | 为何重要                              |
|------------------------------------|---------------------------------------|
| _apply_cot_mode before_answer      | 默认模式；CoT 指标用 boxed 前文本     |
| _apply_cot_mode full_text          | 可选；pred_cot 保留完整生成           |
| _apply_cot_mode 非法值             | 配置 typo 应明确报错                  |
| _split_one_row 正常 / 无 boxed     | answer_row / cot_row 字段结构正确     |
| _split_one_row 缺 generated_text   | infer 产物不完整时应 fail fast        |
| run_split_pipeline 端到端          | 读写 jsonl + extract_rate 统计        |

运行
----
  pip install -r requirements/dev.txt
  pytest tests/test_split.py -v
"""

from __future__ import annotations

import pytest

from src.common.io import read_jsonl
from src.eval.split import _apply_cot_mode, _split_one_row, run_split_pipeline


# ── _apply_cot_mode：pred_cot 写哪段文本 ─────────────────────────────────────


def test_apply_cot_mode_before_answer():
    """默认：pred_cot 为 \\boxed 之前的推理部分（由 extract 抽出）。"""
    generated = "Step 1.\nTherefore \\boxed{12}"
    cot = "Step 1.\nTherefore"

    assert _apply_cot_mode(generated, cot, "before_answer") == cot


def test_apply_cot_mode_full_text():
    """full_text：pred_cot 为整段 generated_text（strip 后）。"""
    generated = "  Step 1.\nTherefore \\boxed{12}  "
    cot = "Step 1.\nTherefore"

    assert _apply_cot_mode(generated, cot, "full_text") == generated.strip()


def test_apply_cot_mode_invalid():
    """未知 cot_mode 应抛 ValueError。"""
    with pytest.raises(ValueError, match="Unsupported split.cot_mode"):
        _apply_cot_mode("text", "cot", "middle_only")


# ── _split_one_row：单行 prediction → answer / cot 两条记录 ───────────────────


def test_split_one_row_with_boxed():
    """有 \\boxed 时：pred_answer、extract_ok、strategy 字段正确。"""
    row = {
        "id": "ex-1",
        "problem": "1+1=?",
        "generated_text": "Compute: 1+1=2. Answer: \\boxed{2}",
    }

    answer_row, cot_row = _split_one_row(row, ["boxed"], "before_answer")

    assert answer_row == {
        "id": "ex-1",
        "problem": "1+1=?",
        "pred_answer": "2",
        "extract_ok": True,
        "strategy": "boxed",
    }
    assert cot_row["pred_cot"] == "Compute: 1+1=2. Answer:"
    assert cot_row["extract_ok"] is True


def test_split_one_row_full_text_cot_mode():
    """full_text 模式下 pred_cot 应含 \\boxed 整段。"""
    row = {
        "id": "ex-2",
        "generated_text": "Reason \\boxed{7}",
    }

    _, cot_row = _split_one_row(row, ["boxed"], "full_text")

    assert cot_row["pred_cot"] == "Reason \\boxed{7}"


def test_split_one_row_no_boxed():
    """无 boxed：extract_ok=False，pred_answer=None。"""
    row = {
        "id": "ex-3",
        "generated_text": "I guess 7",
    }

    answer_row, cot_row = _split_one_row(row, ["boxed"], "before_answer")

    assert answer_row["pred_answer"] is None
    assert answer_row["extract_ok"] is False
    assert answer_row["strategy"] is None
    assert cot_row["pred_cot"] == "I guess 7"


def test_split_one_row_missing_generated_text():
    """prediction 行缺少 generated_text 应抛 KeyError。"""
    with pytest.raises(KeyError, match="Missing 'generated_text'"):
        _split_one_row({"id": "bad"}, ["boxed"], "before_answer")


# ── run_split_pipeline：批量 split + extract_rate ─────────────────────────────


def test_run_split_pipeline_end_to_end(tmp_path):
    """读 predictions jsonl → 写 answer/cot 文件，并正确统计 extract_rate。"""
    pred_path = tmp_path / "pred.jsonl"
    answer_path = tmp_path / "answers.jsonl"
    cot_path = tmp_path / "cots.jsonl"

    # JSON 里 \b 是退格转义；LaTeX \boxed 须写成 \\boxed
    pred_path.write_text(
        "\n".join(
            [
                r'{"id": "1", "problem": "p1", "generated_text": "CoT \\boxed{1}"}',
                '{"id": "2", "problem": "p2", "generated_text": "no marker"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = {
        "split": {
            "prediction_file": str(pred_path),
            "answer_file": str(answer_path),
            "cot_file": str(cot_path),
            "strategies": ["boxed"],
            "cot_mode": "before_answer",
        }
    }

    result = run_split_pipeline(config)

    assert result["rows"] == 2
    assert result["extract_ok_count"] == 1
    assert result["extract_rate"] == 0.5
    assert result["answer_rows_written"] == 2
    assert result["cot_rows_written"] == 2

    answers = list(read_jsonl(answer_path))
    cots = list(read_jsonl(cot_path))

    assert answers[0]["pred_answer"] == "1"
    assert answers[0]["extract_ok"] is True
    assert answers[1]["extract_ok"] is False
    assert cots[0]["pred_cot"] == "CoT"
    assert cots[1]["pred_cot"] == "no marker"


def test_run_split_pipeline_missing_prediction_file(tmp_path):
    """prediction 文件不存在时应抛 FileNotFoundError。"""
    config = {
        "split": {
            "prediction_file": str(tmp_path / "missing.jsonl"),
            "answer_file": str(tmp_path / "a.jsonl"),
            "cot_file": str(tmp_path / "c.jsonl"),
        }
    }

    with pytest.raises(FileNotFoundError, match="Prediction file not found"):
        run_split_pipeline(config)
