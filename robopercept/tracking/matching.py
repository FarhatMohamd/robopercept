"""Assignment helpers for tracking-by-detection."""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except ImportError:  # pragma: no cover - scipy is a hard dependency, fallback kept for safety
    _lsa = None


def linear_assignment(
    cost: np.ndarray, thresh: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Minimum-cost matching, rejecting pairs whose cost exceeds ``thresh``.

    Returns ``(matches, unmatched_rows, unmatched_cols)``.
    """
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    if _lsa is not None:
        padded = np.where(cost > thresh, thresh + 1e4, cost)
        r_idx, c_idx = _lsa(padded)
        pairs = [(int(r), int(c)) for r, c in zip(r_idx, c_idx) if cost[r, c] <= thresh]
    else:  # greedy fallback
        pairs, used_r, used_c = [], set(), set()
        for flat in np.argsort(cost, axis=None):
            r, c = divmod(int(flat), cols)
            if cost[r, c] > thresh:
                break
            if r not in used_r and c not in used_c:
                pairs.append((r, c))
                used_r.add(r)
                used_c.add(c)
    matched_r = {r for r, _ in pairs}
    matched_c = {c for _, c in pairs}
    return (
        pairs,
        [r for r in range(rows) if r not in matched_r],
        [c for c in range(cols) if c not in matched_c],
    )


def fuse_score(iou_cost: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """ByteTrack score fusion: confident detections get a lower matching cost."""
    if iou_cost.size == 0:
        return iou_cost
    sim = (1.0 - iou_cost) * np.asarray(scores, dtype=np.float32)[None, :]
    return 1.0 - sim


def class_penalty(track_cls: np.ndarray, det_cls: np.ndarray, penalty: float = 0.25) -> np.ndarray:
    """Soft cost added when a track and a detection disagree on the class."""
    return penalty * (np.asarray(track_cls)[:, None] != np.asarray(det_cls)[None, :])
