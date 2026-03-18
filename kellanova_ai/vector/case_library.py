"""
Vector Case Library using FAISS + SentenceTransformers
Stores historical resolution cases and retrieves the most similar
cases for a given store issue description.
"""
from __future__ import annotations
import json
import pickle
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config.settings import VECTOR_DIR

INDEX_PATH  = VECTOR_DIR / "case_library.faiss"
META_PATH   = VECTOR_DIR / "case_library_meta.pkl"
MODEL_NAME  = "all-MiniLM-L6-v2"

_encoder: SentenceTransformer | None = None
_index:   faiss.IndexFlatL2   | None = None
_meta:    list[dict]                  = []


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(MODEL_NAME)
    return _encoder


def _case_to_text(case: dict) -> str:
    """Combine case fields into a single searchable string."""
    return (
        f"Issue: {case.get('issue_type','')}. "
        f"Description: {case.get('description','')}. "
        f"Action: {case.get('action_taken','')}. "
        f"Result: {case.get('result','')}."
    )


def build_case_index(case_library: pd.DataFrame) -> None:
    """Encode all cases and build a FAISS flat L2 index."""
    global _index, _meta

    encoder = _get_encoder()
    cases   = case_library.to_dict("records")
    texts   = [_case_to_text(c) for c in cases]
    vectors = encoder.encode(texts, show_progress_bar=False).astype("float32")

    # L2 normalise → cosine similarity via dot product
    faiss.normalize_L2(vectors)

    dim    = vectors.shape[1]
    index  = faiss.IndexFlatIP(dim)        # inner product on normalised = cosine sim
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(cases, f)

    _index = index
    _meta  = cases
    print(f"  Case library index built — {len(cases)} cases, dim={dim}")


def load_case_index() -> bool:
    """Load persisted FAISS index. Returns True if successful."""
    global _index, _meta
    if INDEX_PATH.exists() and META_PATH.exists():
        _index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            _meta = pickle.load(f)
        return True
    return False


def retrieve_similar_cases(
    query: str,
    issue_type: str | None = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve top-k most similar historical cases for a given query string.
    Optionally filter by issue_type before semantic search.
    """
    global _index, _meta

    if _index is None:
        if not load_case_index():
            return []

    encoder = _get_encoder()
    qvec    = encoder.encode([query], show_progress_bar=False).astype("float32")
    faiss.normalize_L2(qvec)

    # Search larger pool then filter
    search_k = min(len(_meta), top_k * 5)
    scores, idxs = _index.search(qvec, search_k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0 or idx >= len(_meta):
            continue
        case = dict(_meta[idx])
        if issue_type and case.get("issue_type") != issue_type:
            continue
        case["similarity_score"] = round(float(score), 4)
        results.append(case)
        if len(results) >= top_k:
            break

    # Fallback: if filtered list is empty, return unfiltered top-k
    if not results:
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(_meta):
                continue
            case = dict(_meta[idx])
            case["similarity_score"] = round(float(score), 4)
            results.append(case)
            if len(results) >= top_k:
                break

    return results


def format_cases_for_display(cases: list[dict]) -> pd.DataFrame:
    if not cases:
        return pd.DataFrame(columns=["case_id","issue_type","description","action_taken","result","similarity_score"])
    df = pd.DataFrame(cases)
    return df[["case_id","issue_type","description","action_taken","result","similarity_score"]]

