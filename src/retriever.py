import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from .config import RRF_K, TOP_K_BM25, TOP_K_FINAL, TOP_K_VECTOR
from .utils import chunk_key, unique_by_key


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_index(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def vector_search(query: str, vectorstore, top_k: int = TOP_K_VECTOR) -> List[Dict[str, Any]]:
    docs = vectorstore.similarity_search(query, k=top_k)

    results = []
    for rank, doc in enumerate(docs, start=1):
        meta = dict(doc.metadata or {})
        meta["chunk_id"] = meta.get("chunk_id") or chunk_key(meta)
        results.append(
            {
                "key": meta["chunk_id"],
                "content": doc.page_content,
                "metadata": meta,
                "retrieval_sources": ["vector"],
                "vector_rank": rank,
                "bm25_rank": None,
                "rrf_score": 1 / (RRF_K + rank),
            }
        )
    return results


def bm25_search(query: str, chunks: List[Dict[str, Any]], bm25: BM25Okapi, top_k: int = TOP_K_BM25) -> List[Dict[str, Any]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        chunk = chunks[idx]
        meta = dict(chunk["metadata"])
        meta["chunk_id"] = meta.get("chunk_id") or chunk_key(meta)

        results.append(
            {
                "key": meta["chunk_id"],
                "content": chunk["text"],
                "metadata": meta,
                "retrieval_sources": ["bm25"],
                "vector_rank": None,
                "bm25_rank": rank,
                "rrf_score": 1 / (RRF_K + rank),
                "bm25_score": float(scores[idx]),
            }
        )
    return results


def hybrid_search(
    query: str,
    vectorstore,
    chunks: List[Dict[str, Any]],
    bm25: BM25Okapi,
    top_k_vector: int = TOP_K_VECTOR,
    top_k_bm25: int = TOP_K_BM25,
    final_k: int = TOP_K_FINAL,
) -> List[Dict[str, Any]]:
    vector_results = vector_search(query, vectorstore, top_k=top_k_vector)
    bm25_results = bm25_search(query, chunks, bm25, top_k=top_k_bm25)

    combined = {}

    def add_result(item: Dict[str, Any], source_type: str, rank_value: int):
        key = item["key"]
        if key not in combined:
            combined[key] = {
                "key": key,
                "content": item["content"],
                "metadata": item["metadata"],
                "retrieval_sources": [],
                "vector_rank": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
                "bm25_score": None,
            }

        combined[key]["rrf_score"] += 1 / (RRF_K + rank_value)
        if source_type not in combined[key]["retrieval_sources"]:
            combined[key]["retrieval_sources"].append(source_type)

        if source_type == "vector":
            combined[key]["vector_rank"] = rank_value
        else:
            combined[key]["bm25_rank"] = rank_value
            combined[key]["bm25_score"] = item.get("bm25_score")

    for item in vector_results:
        add_result(item, "vector", item["vector_rank"])

    for item in bm25_results:
        add_result(item, "bm25", item["bm25_rank"])

    ranked = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
    ranked = unique_by_key(ranked, key_field="key")
    return ranked[:final_k]
