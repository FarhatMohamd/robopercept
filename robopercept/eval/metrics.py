"""Evaluation against ground truth: detection, tracking (CLEAR-MOT) and ranging.

The simulator provides exact ground truth, so the whole stack can be scored on
every commit. The same functions work on any dataset that provides per-frame
boxes and identities (KITTI tracking, BDD100K MOT, MOT17...).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..detectors.nms import iou_matrix
from ..tracking.matching import linear_assignment


@dataclass
class MotAccumulator:
    """CLEAR-MOT counters (Bernardin & Stiefelhagen, 2008) and ID-based scores."""

    iou_thresh: float = 0.5
    num_gt: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    id_switches: int = 0
    iou_sum: float = 0.0
    _last_match: dict[int, int] = field(default_factory=dict)
    _pair_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    _gt_counts: dict[int, int] = field(default_factory=dict)
    _hyp_counts: dict[int, int] = field(default_factory=dict)
    range_errors: list[tuple[float, float, str]] = field(default_factory=list)

    def update(
        self,
        gt_ids: list[int],
        gt_boxes: np.ndarray,
        hyp_ids: list[int],
        hyp_boxes: np.ndarray,
        gt_ranges: list[float] | None = None,
        hyp_ranges: list[float | None] | None = None,
        gt_classes: list[str] | None = None,
    ) -> list[tuple[int, int]]:
        self.num_gt += len(gt_ids)
        for g in gt_ids:
            self._gt_counts[g] = self._gt_counts.get(g, 0) + 1
        for h in hyp_ids:
            self._hyp_counts[h] = self._hyp_counts.get(h, 0) + 1
        if len(gt_ids) == 0 or len(hyp_ids) == 0:
            self.fn += len(gt_ids)
            self.fp += len(hyp_ids)
            return []
        ious = iou_matrix(np.asarray(gt_boxes), np.asarray(hyp_boxes))
        # Prefer keeping previous correspondences (standard CLEAR-MOT rule)
        cost = 1.0 - ious
        for gi, g in enumerate(gt_ids):
            prev = self._last_match.get(g)
            if prev is not None and prev in hyp_ids:
                hj = hyp_ids.index(prev)
                if ious[gi, hj] >= self.iou_thresh:
                    cost[gi, hj] -= 1.0
        matches, u_gt, u_hyp = linear_assignment(cost, 1.0 - self.iou_thresh)
        self.tp += len(matches)
        self.fn += len(u_gt)
        self.fp += len(u_hyp)
        for gi, hj in matches:
            g, h = gt_ids[gi], hyp_ids[hj]
            self.iou_sum += float(ious[gi, hj])
            prev = self._last_match.get(g)
            if prev is not None and prev != h:
                self.id_switches += 1
            self._last_match[g] = h
            self._pair_counts[(g, h)] = self._pair_counts.get((g, h), 0) + 1
            if gt_ranges is not None and hyp_ranges is not None and hyp_ranges[hj] is not None:
                cls = gt_classes[gi] if gt_classes else ""
                self.range_errors.append((float(gt_ranges[gi]), float(hyp_ranges[hj]), cls))
        return [(gt_ids[gi], hyp_ids[hj]) for gi, hj in matches]

    def idf1(self) -> float:
        """ID-F1 using a global bipartite matching of identities (Ristani et al., 2016)."""
        gts = sorted(self._gt_counts)
        hyps = sorted(self._hyp_counts)
        if not gts or not hyps:
            return 0.0
        overlap = np.zeros((len(gts), len(hyps)))
        gi = {g: i for i, g in enumerate(gts)}
        hi = {h: j for j, h in enumerate(hyps)}
        for (g, h), n in self._pair_counts.items():
            overlap[gi[g], hi[h]] = n
        matches, _, _ = linear_assignment(-overlap, 0.0)
        idtp = sum(overlap[i, j] for i, j in matches)
        total_gt = sum(self._gt_counts.values())
        total_hyp = sum(self._hyp_counts.values())
        return float(2 * idtp / max(total_gt + total_hyp, 1))

    def summary(self) -> dict[str, float]:
        mota = 1.0 - (self.fn + self.fp + self.id_switches) / max(self.num_gt, 1)
        out = {
            "MOTA": mota,
            "MOTP_iou": self.iou_sum / max(self.tp, 1),
            "IDF1": self.idf1(),
            "precision": self.tp / max(self.tp + self.fp, 1),
            "recall": self.tp / max(self.tp + self.fn, 1),
            "id_switches": float(self.id_switches),
            "gt_objects": float(self.num_gt),
        }
        out.update(range_metrics(self.range_errors))
        return out


def range_metrics(pairs: list[tuple[float, float, str]]) -> dict[str, float]:
    """Absolute relative error of the range, overall and per distance band."""
    if not pairs:
        return {}
    gt = np.array([p[0] for p in pairs])
    est = np.array([p[1] for p in pairs])
    rel = np.abs(est - gt) / np.maximum(gt, 0.5)
    out = {
        "range_absrel_mean": float(rel.mean()),
        "range_absrel_median": float(np.median(rel)),
        "range_abs_mae_m": float(np.abs(est - gt).mean()),
    }
    for lo, hi in ((0, 10), (10, 30), (30, 80)):
        sel = (gt >= lo) & (gt < hi)
        if sel.any():
            out[f"range_absrel_{lo}-{hi}m"] = float(rel[sel].mean())
    return out
