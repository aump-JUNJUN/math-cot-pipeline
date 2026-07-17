from __future__ import annotations
import numpy as np

"""
 ParlAI中的informativeness_chain指标
"""

def cosine_sim_01(a: np.ndarray, b: np.ndarray) -> float:
    """归一化向量的 cosine，映射到 [0, 1]。"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    return (1.0 + cos) / 2.0


class RoscoEmbedder:
    _cache: dict[str, "RoscoEmbedder"] = {}

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self._model = None

    @classmethod
    def get(cls, name: str) -> "RoscoEmbedder":
        if name not in cls._cache:
            cls._cache[name] = cls(name)
        return cls._cache[name]

    def encode(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(1)
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec)


def informativeness_chain_score(
    problem: str,
    pred_cot: str,
    embedder: RoscoEmbedder,
) -> float:
    """主函数：ROSCOE informativeness_chain：整段 pred vs 整段 problem。"""
    if not (pred_cot or "").strip():
        return 0.0
    e_pred = embedder.encode(pred_cot)
    e_ctx = embedder.encode(problem)
    return cosine_sim_01(e_pred, e_ctx)


if __name__ == "__main__":
    # 1) 不加载模型
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert cosine_sim_01(a, b) == 1.0

    embedder = RoscoEmbedder.get("all-mpnet-base-v2")
    assert informativeness_chain_score("Some problem.", "", embedder) == 0.0
    print("cosine + empty pred: ok")

    # 2) 一条样本（需要 sentence-transformers）
    problem = "Solve x^2 = 4 for x."
    pred_cot = "We have x^2 = 4. Taking square roots gives x = 2 or x = -2."
    score = informativeness_chain_score(problem, pred_cot, embedder)
    print(f"informativeness_chain (one sample): {score:.4f}")
    assert 0.0 <= score <= 1.0
    print("roscoe_utils smoke test passed.")
