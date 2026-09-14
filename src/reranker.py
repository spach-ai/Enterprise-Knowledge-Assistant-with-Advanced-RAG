from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL_NAME, TOP_K_FINAL


class DocumentReranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, cand["content"]) for cand in candidates]
        scores = self.model.predict(pairs)

        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
