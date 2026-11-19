"""Score the full pipeline on a simulator scenario with exact ground truth."""

from __future__ import annotations

import numpy as np

from ..config import Config
from .metrics import MotAccumulator


def evaluate_synthetic(cfg: Config, frames: int = 900) -> dict[str, float]:
    """Run ``frames`` frames and return tracking, ranging and latency metrics.

    Range is compared as the longitudinal distance ``z`` of the nearest face,
    which is what the monocular cues measure.
    """
    from ..pipeline import Pipeline

    cfg = cfg.model_copy(deep=True)
    if not cfg.source.uri.startswith("synthetic"):
        cfg.source.uri = "synthetic:urban"
    cfg.source.max_frames = frames
    cfg.output.show = False
    cfg.server.enabled = False
    pipe = Pipeline(cfg)
    acc = MotAccumulator()
    lateral: list[float] = []
    levels: list[int] = []
    for res, _ in pipe.stream():
        gt = res.meta.get("ground_truth", [])
        gt_boxes = np.array([g["box"] for g in gt]).reshape(-1, 4)
        hyp_boxes = np.array([t.box for t in res.tracks]).reshape(-1, 4)
        matches = acc.update(
            [g["gt_id"] for g in gt],
            gt_boxes,
            [t.track_id for t in res.tracks],
            hyp_boxes,
            gt_ranges=[g["z"] for g in gt],
            hyp_ranges=[t.world.z if t.world else None for t in res.tracks],
            gt_classes=[g["class_name"] for g in gt],
        )
        by_gt = {g["gt_id"]: g for g in gt}
        by_hyp = {t.track_id: t for t in res.tracks}
        for g_id, h_id in matches:
            tr = by_hyp[h_id]
            if tr.world is not None:
                lateral.append(abs(tr.world.x - by_gt[g_id]["x"]))
        levels.append(int(res.decision.level))
    out = acc.summary()
    if lateral:
        out["lateral_mae_m"] = float(np.mean(lateral))
    lat = pipe.timer.summary().get("total", {})
    out["latency_mean_ms"] = lat.get("mean", float("nan"))
    out["latency_p95_ms"] = lat.get("p95", float("nan"))
    lv = np.array(levels)
    out["share_warning_or_above"] = float((lv >= 2).mean()) if lv.size else 0.0
    return out
