"""
answer和cot分别计算指标
仅仅将指标的包安放在此，本文件只定义指标类，调度见 run_metrics.py。
"""

from src.common.registry import register_answer_metric, register_cot_metric
from typing import Dict, List, Union
from abc import ABC, abstractmethod


def normalize_text(text: str) -> str:
    """Normalize text by lowering case and stripping whitespace."""
    return text.strip().lower()


class Metric(ABC):
    """Sample-level metric: apply(preds, refs) -> per-sample scores."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        pass

    def aggregate(self, scores: List[float]) -> float:
        """Default: mean over samples."""
        return sum(scores) / len(scores) if scores else 0.0

    def __call__(self, prediction: str, reference: str) -> float:
        return self.apply([prediction], [reference])[0]


@register_answer_metric('exact_match')
class ExactMatch(Metric):

    def apply(self, predictions, references):
        return [
            float(normalize_text(prediction) == normalize_text(reference))
            for prediction, reference in zip(predictions, references)
        ]


@register_answer_metric('acc')
class Accuracy(ExactMatch):

    def __init__(self, allow_inclusion: bool = False, numeric: bool = False):
        self.allow_inclusion = allow_inclusion
        self.numeric = numeric

    def apply(self, predictions, references):
        if self.allow_inclusion:
            results = []
            for prediction, reference in zip(predictions, references):
                if prediction and prediction in reference:
                    results.append(1.0)
                else:
                    results.append(0.0)
            return results
        elif self.numeric:
            from evalscope.metrics.math_parser import math_equal, strip_answer_string

            results = []
            for prediction, reference in zip(predictions, references):
                ref_answer = strip_answer_string(reference)
                results.append(float(math_equal(prediction, ref_answer)))

            return results
        else:
            return super().apply(predictions, references)


@register_answer_metric('extract_rate')
class ExtractRateMetric:
    def apply_from_rows(self, pred_rows: list[dict]) -> list[float]:
        return [1.0 if row.get("extract_ok") else 0.0 for row in pred_rows]

    def aggregate(self, scores):
        return sum(scores) / len(scores) if scores else 0.0


"""   COT部分的指标  """

#显存溢出 修改batch_size + 精度 bfloat16
@register_cot_metric('bert_score')
class BertScoreMetric:
    def __init__(
        self,
        lang: str = "en",
        model_type: str = "microsoft/deberta-base-mnli",
        batch_size: int = 1,
        device: str = "cuda",
        dtype: str | None = "bfloat16",
    ):
        self.lang = lang
        self.model_type = model_type
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

    def _resolve_autocast_dtype(self):
        import torch

        if not self.dtype:
            return None
        key = self.dtype.lower().replace("-", "")
        if key in ("float32", "fp32", "none", "null"):
            return None
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
        }
        if key not in mapping:
            raise ValueError(f"Unsupported bert_score dtype: {self.dtype!r}")
        return mapping[key]

    def apply(self, predictions, references):
        import torch
        from bert_score import score

        score_kwargs = {
            "lang": self.lang,
            "model_type": self.model_type,
            "batch_size": self.batch_size,
            "device": self.device,
        }
        autocast_dtype = self._resolve_autocast_dtype()
        use_autocast = (
            autocast_dtype is not None
            and str(self.device).startswith("cuda")
            and torch.cuda.is_available()
        )

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                P, R, F1 = score(predictions, references, **score_kwargs)
        else:
            P, R, F1 = score(predictions, references, **score_kwargs)

        return F1.tolist()  # 每条样本一个 F1

    def aggregate(self, scores):
        return sum(scores) / len(scores) if scores else 0.0



from src.eval.roscoe_utils import RoscoEmbedder, informativeness_chain_score


@register_cot_metric("informativeness_chain")
class InformativenessChainMetric:
    """整段 pred_cot vs problem（ROSCOE informativeness_chain，极简版）。"""

    def __init__(self, embedding_model: str = "all-mpnet-base-v2"):
        self._embedder = RoscoEmbedder.get(embedding_model)

    def apply_with_context(
        self,
        rows: list[dict],
        pred_field: str = "pred_cot",
        context_field: str = "problem",
    ) -> list[float]:
        scores: list[float] = []
        for row in rows:
            pred_cot = row.get(pred_field) or ""
            problem = row.get(context_field) or ""
            scores.append(
                informativeness_chain_score(problem, pred_cot, self._embedder)
            )
        return scores

    def aggregate(self, scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0


        