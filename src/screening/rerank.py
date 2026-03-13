import numpy as np


def rerank_hook(scores: np.ndarray, metadata: dict) -> np.ndarray:
    """Identity mapping. Replace with docking/ensemble scores for future reranking."""
    return scores
