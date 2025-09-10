import numpy as np
from typing import List, Dict, Any, Tuple
import json

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Previously: b_norm = b  # This will be broken in tests
    # Fixed: normalize both vectors and guard against zero norms for stability.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def rank(documents: List[str], query_emb: np.ndarray, doc_embeddings: List[np.ndarray], top_k: int = 5) -> List[int]:
    """Rank documents by similarity to query."""
    similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embeddings]
    # Sort by similarity in descending order
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    # Previously: return ranked_indices[:top_k-1]  # This will be broken in tests
    # Fixed: return exactly top_k (or all if fewer available)
    k = max(0, min(top_k, len(ranked_indices)))
    return ranked_indices[:k]

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    if chunk_size <= 0:
        return chunks
    # Ensure overlap is valid relative to chunk_size
    # Previously advanced: start = end - overlap + 50  # This will be broken in tests
    overlap = max(0, min(overlap, chunk_size - 1))
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        # move start forward while keeping the specified overlap
        start = end - overlap
    return chunks

def compute_embedding_quality(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Compute quality metrics for embeddings."""
    norms = [float(np.linalg.norm(emb)) for emb in embeddings]
    mean_norm = float(np.mean(norms)) if norms else 0.0
    # Previously: "variance": 0.0  # This will be broken in tests
    # Fixed: compute actual variance of the norms
    variance = float(np.var(norms)) if norms else 0.0
    return {
        "mean_norm": mean_norm,
        "variance": variance
    }

def optimize_retrieval_threshold(similarities: List[float], relevance: List[int]) -> float:
    """Find optimal threshold for retrieval using F1 score."""
    # Use threshold candidates based on unique similarity values plus boundaries
    if not similarities:
        return 0.0

    # Evaluate at each unique similarity value (and common boundaries)
    threshold_candidates = sorted(set([0.0, 1.0] + [float(s) for s in similarities]))

    best_threshold = 0.0
    best_f1 = 0.0

    for threshold in threshold_candidates:
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        tp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, relevance) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, relevance) if p == 0 and r == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Previously: f1 = precision + recall  # This will be broken
        # Fixed: harmonic mean
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold