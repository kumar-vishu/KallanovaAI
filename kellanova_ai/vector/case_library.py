"""
Vector Case Library — lightweight TF-IDF retrieval (no PyTorch/FAISS required).
Uses sklearn TfidfVectorizer + numpy cosine similarity so the app deploys fast
on Streamlit Community Cloud without pulling in torch (~1.5 GB).
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import VECTOR_DIR

# Persist the fitted vectorizer + matrix so we only build once
INDEX_PATH = VECTOR_DIR / "case_library_tfidf.pkl"   # replaces .faiss
META_PATH  = VECTOR_DIR / "case_library_meta.pkl"

_vectorizer: TfidfVectorizer | None = None
_matrix:     np.ndarray      | None = None   # shape (n_cases, n_features)
_meta:       list[dict]             = []


def _case_to_text(case: dict) -> str:
    """Combine case fields into a single searchable string."""
    return (
        f"Issue: {case.get('issue_type','')}. "
        f"Description: {case.get('description','')}. "
        f"Action: {case.get('action_taken','')}. "
        f"Result: {case.get('result','')}."
    )


def build_case_index(case_library: pd.DataFrame) -> None:
    """Fit TF-IDF on all cases and persist the vectorizer + matrix."""
    global _vectorizer, _matrix, _meta

    cases  = case_library.to_dict("records")
    texts  = [_case_to_text(c) for c in cases]

    vec    = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
    matrix = vec.fit_transform(texts).toarray().astype("float32")

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"vectorizer": vec, "matrix": matrix}, f)
    with open(META_PATH, "wb") as f:
        pickle.dump(cases, f)

    _vectorizer, _matrix, _meta = vec, matrix, cases
    print(f"  Case library index built — {len(cases)} cases (TF-IDF)")


def load_case_index() -> bool:
    """Load persisted TF-IDF index. Returns True if successful."""
    global _vectorizer, _matrix, _meta
    if INDEX_PATH.exists() and META_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            obj = pickle.load(f)
        _vectorizer = obj["vectorizer"]
        _matrix     = obj["matrix"]
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
    Optionally filter by issue_type.
    """
    global _vectorizer, _matrix, _meta

    if _vectorizer is None:
        if not load_case_index():
            return []

    qvec   = _vectorizer.transform([query]).toarray().astype("float32")
    scores = cosine_similarity(qvec, _matrix)[0]   # shape (n_cases,)

    ranked = np.argsort(scores)[::-1]

    results = []
    for idx in ranked:
        case = dict(_meta[idx])
        if issue_type and case.get("issue_type") != issue_type:
            continue
        case["similarity_score"] = round(float(scores[idx]), 4)
        results.append(case)
        if len(results) >= top_k:
            break

    # Fallback: return unfiltered top-k if filter gave nothing
    if not results:
        for idx in ranked[:top_k]:
            case = dict(_meta[idx])
            case["similarity_score"] = round(float(scores[idx]), 4)
            results.append(case)

    return results


def format_cases_for_display(cases: list[dict]) -> pd.DataFrame:
    if not cases:
        return pd.DataFrame(columns=["case_id","issue_type","description","action_taken","result","similarity_score"])
    df = pd.DataFrame(cases)
    return df[["case_id","issue_type","description","action_taken","result","similarity_score"]]

