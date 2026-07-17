from __future__ import annotations

"""
评测流水线第 1 步：批量推理（需 GPU）。

读 configs/eval_ft.yaml 或 eval_base.yaml → infer 块 + data/processed/test.jsonl，
调用 ms-swift 对 test 集逐条生成，结果写入 reports/predictions/。

整条 eval 链路：
  infer.py       → reports/predictions/{name}.jsonl          （本文件）
  split.py       → reports/predictions/{name}_answers.jsonl   （抽 answer / cot）
                   reports/predictions/{name}_cots.jsonl
  run_metrics.py → reports/metrics/{name}_answer.json         （聚合指标）
                   reports/metrics/{name}_cot.json
  report.py      → reports/metrics/compare_all.json + .png   （读 configs/compare.yaml）

本文件只写 predictions/，不写 metrics/。
输出每行仅含：id, problem, generated_text, model_path, model_id
（不含 pred_answer / pred_cot，留给 split.py）

模型加载：
  - ft：configs/eval_ft.yaml，model_path → outputs/merged/best（需先 export）；本地路径需配 model_type
  - base：configs/eval_base.yaml，model_path 设为 null，回退到 model_id

启动：
  python -m src.eval.infer
  python -m src.eval.infer --config configs/eval_base.yaml
  python -m src.eval.infer --limit 4
"""

import argparse
import copy
import os
import re
from pathlib import Path
from typing import Any

from src.common.config import load_config
from src.common.io import read_jsonl, write_jsonl

# 匹配 infer_schema 里的占位符，如 <problem>
_PLACEHOLDER_RE = re.compile(r"<(\w+)>")


def _setup_runtime(infer_cfg: dict[str, Any]) -> None:
    """按 infer.runtime 设置 CUDA_VISIBLE_DEVICES；须在 import torch / ms-swift 之前调用。"""
    runtime = infer_cfg.get("runtime", {})
    cuda_devices = runtime.get("cuda_visible_devices")
    if cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)


def _fill_placeholders(text: str, row: dict[str, Any]) -> str:
    """把模板字符串中的 <field> 替换为 row[field] 的字符串形式。"""
    def replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        value = row.get(key)
        if value is None:
            return match.group(0)
        return str(value)

    return _PLACEHOLDER_RE.sub(replacer, text)


def _apply_infer_schema(row: dict[str, Any], infer_schema: dict[str, Any]) -> dict[str, Any]:
    """
    将单行 test 样本按 infer_schema 转为 ms-swift 所需的 messages 结构。
    规则与 src/train/format.py 一致。
    """
    schema_copy = copy.deepcopy(infer_schema)
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


def _build_messages(row: dict[str, Any], infer_cfg: dict[str, Any]) -> list[dict[str, str]]:
    """构造单条 InferRequest 的 messages；若有 system_prompt 则插在最前面。"""
    formatted = _apply_infer_schema(row, infer_cfg["infer_schema"])["messages"]
    system_prompt = infer_cfg.get("system_prompt")
    if system_prompt:
        return [{"role": "system", "content": str(system_prompt)}] + formatted
    return formatted


def _resolve_model_source(infer_cfg: dict[str, Any]) -> str:
    """
    解析实际加载路径：
      - model_path 有值且目录存在 → merged 本地模型
      - model_path 为 null / 空     → 回退 HuggingFace model_id（base 评测）
    """
    model_path = infer_cfg.get("model_path")
    model_id = infer_cfg["model_id"]

    if model_path is None or str(model_path).strip().lower() in ("", "null", "none"):
        return model_id

    resolved = Path(os.path.expanduser(str(model_path))).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Model path not found: {resolved}. "
            "Run `python -m src.train.export` first, or set model_path: null for base model."
        )
    return str(resolved)


def _import_infer_api() -> tuple[Any, Any, Any]:
    """
    工程化代码：兼容不同版本的 ms-swift 导入路径。
    作用：导入 ms-swift 推理相关的 API，包括 Engine 类、RequestConfig 配置类、InferRequest 请求类型。
    为什么要兼容？因为 ms-swift 不同版本下这些类的导入路径有差异，所以需要逐一尝试，用于适应不同的包结构，保证在不同版本下 eval 脚本都能正常运行。
    """
    try:
        from swift.llm import InferRequest, PtEngine, RequestConfig
        return PtEngine, RequestConfig, InferRequest
    except ImportError:
        pass

    try:
        # 较早版本的导入路径
        from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
        return TransformersEngine, RequestConfig, InferRequest
    except ImportError:
        pass

    try:
        # 更旧或备用的导入路径
        from swift import InferRequest, RequestConfig, TransformersEngine
        return TransformersEngine, RequestConfig, InferRequest
    except ImportError as exc:
        # 以上都失败，说明 ms-swift 未正确安装
        raise ImportError(
            "ms-swift inference API not found. Install with: pip install -r requirements/train.txt"
        ) from exc


def _load_engine(
    model_source: str,
    max_batch_size: int,
    model_type: str | None = None,
) -> Any:
    """加载 PtEngine 或 TransformersEngine；本地 merged 模型需传 model_type。"""
    EngineCls, _, _ = _import_infer_api()
    kwargs: dict[str, Any] = {"max_batch_size": max_batch_size}
    if model_type:
        kwargs["model_type"] = model_type
    return EngineCls(model_source, **kwargs)


def _build_request_config(infer_cfg: dict[str, Any]) -> Any:
    """从配置 infer 块的 temperature / max_new_tokens 构造 RequestConfig。"""
    _, RequestConfig, _ = _import_infer_api()
    return RequestConfig(
        max_tokens=int(infer_cfg.get("max_new_tokens", 1024)),
        temperature=float(infer_cfg.get("temperature", 0.0)),
    )


def _response_text(resp: Any) -> str:
    """从单条 infer 响应中提取 assistant 生成文本。"""
    content = resp.choices[0].message.content
    return content if content is not None else ""


def _build_prediction_row(
    row: dict[str, Any],
    generated_text: str,
    model_source: str,
    model_id: str,
) -> dict[str, Any]:
    """组装写入 predictions/*.jsonl 的单行记录。"""
    return {
        "id": row.get("id"),
        "problem": row.get("problem"),
        "generated_text": generated_text,
        "model_path": model_source,
        "model_id": model_id,
    }


def _iter_test_rows(infer_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """读取 test_file；infer_limit 非 null 时只取前 N 条（快速试跑）。"""
    test_path = Path(infer_cfg["test_file"])
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    rows = list(read_jsonl(test_path))
    limit = infer_cfg.get("infer_limit")
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def run_infer_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """
    主流程：读 test → 批量推理 → 写 prediction_file。

    推理 expensive，结果落盘后可被 split / metrics 反复读取，无需重跑 GPU。
    """
    infer_cfg = config["infer"]
    _setup_runtime(infer_cfg)

    from swift.utils import get_logger

    _, _, InferRequest = _import_infer_api()
    logger = get_logger()

    model_source = _resolve_model_source(infer_cfg)
    model_id = infer_cfg["model_id"]
    max_batch_size = int(infer_cfg.get("max_batch_size", 1))
    prediction_path = Path(infer_cfg["prediction_file"])
    test_rows = _iter_test_rows(infer_cfg)
    request_config = _build_request_config(infer_cfg)

    logger.info(f"model_id: {model_id}")
    logger.info(f"model_source: {model_source}")
    if infer_cfg.get("model_type"):
        logger.info(f"model_type: {infer_cfg['model_type']}")
    logger.info(f"test_file: {infer_cfg['test_file']}")
    logger.info(f"rows: {len(test_rows)}")
    logger.info(f"max_batch_size: {max_batch_size}")
    logger.info(f"prediction_file: {prediction_path}")

    engine = _load_engine(
        model_source,
        max_batch_size=max_batch_size,
        model_type=infer_cfg.get("model_type"),
    )

    predictions: list[dict[str, Any]] = []
    for start in range(0, len(test_rows), max_batch_size):
        batch_rows = test_rows[start : start + max_batch_size]
        infer_requests = [
            # 这里是构建每个样本(batch)的推理请求，InferRequest 类通常用于封装模型推理所需的输入信息。
            # _build_messages(row, infer_cfg) 根据当前样本和推理配置自动生成 prompt 格式的 messages 输入。
            InferRequest(messages=_build_messages(row, infer_cfg))
            for row in batch_rows
        ]
        
        #批量调用API
        resp_list = engine.infer(infer_requests, request_config)

        if len(resp_list) != len(batch_rows):
            raise RuntimeError(
                f"Infer batch size mismatch: got {len(resp_list)} responses "
                f"for {len(batch_rows)} inputs"
            )

        # zip 是 Python 内置函数，用于将多个可迭代对象“并行”组合成元组序列
        # 下面这行的含义是，把 batch_rows 和 resp_list 中的元素一一配对
        for row, resp in zip(batch_rows, resp_list):
            predictions.append(
                _build_prediction_row(row, _response_text(resp), model_source, model_id)
            )

        done = min(start + max_batch_size, len(test_rows))
        logger.info(f"inferred {done}/{len(test_rows)}")

    written = write_jsonl(prediction_path, predictions)

    return {
        "model_id": model_id,
        "model_source": model_source,
        "model_type": infer_cfg.get("model_type"),
        "test_file": infer_cfg["test_file"],
        "prediction_file": str(prediction_path),
        "rows": written,
        "max_batch_size": max_batch_size,
        "temperature": infer_cfg.get("temperature", 0.0),
        "max_new_tokens": infer_cfg.get("max_new_tokens", 1024),
    }


def run_from_config(config_path: str | Path, limit: int | None = None) -> dict[str, Any]:
    """从 yaml 路径加载配置并执行推理；CLI --limit 可覆盖 infer.infer_limit。"""
    config = load_config(config_path)
    if limit is not None:
        config.setdefault("infer", {})["infer_limit"] = limit
    return run_infer_pipeline(config)


def main() -> None:
    """CLI 入口：python -m src.eval.infer"""
    parser = argparse.ArgumentParser(description="Batch inference for eval via ms-swift")
    parser.add_argument("--config", default="configs/eval_ft.yaml", help="评测配置文件路径")
    parser.add_argument("--limit", type=int, default=None, help="覆盖 infer.infer_limit，快速试跑")
    args = parser.parse_args()

    result = run_from_config(args.config, limit=args.limit)
    print("infer done:")
    print(f"  model_source: {result['model_source']}")
    print(f"  rows: {result['rows']}")
    print(f"  prediction_file: {result['prediction_file']}")


if __name__ == "__main__":
    main()

    

