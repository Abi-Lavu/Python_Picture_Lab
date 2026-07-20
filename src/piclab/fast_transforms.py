"""
Vectorized versions of transforms.py, doing the same thing with whole-array
numpy operations instead of explicit pixel loops. Same inputs and outputs as
the loop-based versions -- this module is meant for a side-by-side lesson on
why ML code stops writing for-loops over pixels and starts writing array ops.
"""

import numpy as np


def zero_blue(pixels: np.ndarray) -> np.ndarray:
    result = pixels.copy()
    result[:, :, 2] = 0
    return result


def mirror_vertical(pixels: np.ndarray) -> np.ndarray:
    """Copy the left half onto its mirror position on the right (matches
    transforms.mirror_vertical: the left half itself is left untouched)."""
    result = pixels.copy()
    width = result.shape[1]
    half = width // 2
    result[:, width - half :, :] = pixels[:, :half, :][:, ::-1, :]
    return result


def edge_detection(pixels: np.ndarray, edge_dist: float) -> np.ndarray:
    result = pixels.copy()
    left = pixels[:, :-1, :].astype(np.float32)
    right = pixels[:, 1:, :].astype(np.float32)
    dist = np.sqrt(((left - right) ** 2).sum(axis=2))
    edges = dist > edge_dist
    result[:, :-1, :] = np.where(edges[..., None], 0, 255)
    return result
