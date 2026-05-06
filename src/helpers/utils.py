import hashlib
import re
from collections.abc import Iterable


def file_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    clean_text = normalize_space(text)
    if not clean_text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        chunks.append(clean_text[start:end])
        if end == len(clean_text):
            break
        start = end - chunk_overlap
    return chunks


def lexical_overlap_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    doc_terms = set(re.findall(r"\w+", text.lower()))
    if not doc_terms:
        return 0.0
    return len(query_terms & doc_terms) / len(query_terms)


def rank_fusion(
    semantic_ids: Iterable[str], lexical_ids: Iterable[str], k: int = 60
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(semantic_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for rank, chunk_id in enumerate(lexical_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores
