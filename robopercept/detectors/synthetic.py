"""Ground-truth driven detector for the built-in simulator.

It reproduces the typical failure modes of a real detector (missed objects,
jittered boxes, spurious detections, score spread) so that the whole stack can
be exercised and evaluated without a GPU or downloaded weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import Detection
from .base import Detector

SYNTHETIC_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic cone"]


class SyntheticDetector(Detector):
    name = "synthetic"

    def __init__(
        self,
        conf: float = 0.3,
        classes: list[str] | None = None,
        miss_rate: float = 0.04,
        box_noise_px: float = 1.5,
        false_positive_rate: float = 0.02,
        seed: int = 7,
    ) -> None:
        super().__init__(conf, classes)
        self.miss_rate = miss_rate
        self.box_noise_px = box_noise_px
        self.false_positive_rate = false_positive_rate
        self.rng = np.random.default_rng(seed)

    @property
    def class_names(self) -> list[str]:
        return list(SYNTHETIC_NAMES)

    def _predict(self, image: np.ndarray, meta: dict[str, Any] | None) -> list[Detection]:
        h, w = image.shape[:2]
        out: list[Detection] = []
        for obj in (meta or {}).get("ground_truth", []):
            box = np.asarray(obj["box"], dtype=np.float32)
            visible = obj.get("visibility", 1.0)
            size = max(box[3] - box[1], 1.0)
            # Small and occluded objects are missed more often
            p_miss = self.miss_rate + 0.25 * (1.0 - visible) + (0.15 if size < 14 else 0.0)
            if self.rng.random() < p_miss:
                continue
            noise = self.rng.normal(0.0, self.box_noise_px * (0.5 + size / 200.0), 4)
            box = box + noise.astype(np.float32)
            box[[0, 2]] = box[[0, 2]].clip(0, w - 1)
            box[[1, 3]] = box[[1, 3]].clip(0, h - 1)
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            score = float(np.clip(0.55 + 0.4 * visible + self.rng.normal(0, 0.08), 0.05, 0.99))
            if size < 20:
                score *= 0.75
            out.append(Detection(box, score, SYNTHETIC_NAMES.index(obj["class_name"]), obj["class_name"]))
        if self.rng.random() < self.false_positive_rate * 10:
            # Low-confidence clutter, the kind ByteTrack's second stage must ignore
            cx, cy = self.rng.uniform(0.1 * w, 0.9 * w), self.rng.uniform(0.45 * h, 0.9 * h)
            bw, bh = self.rng.uniform(15, 60), self.rng.uniform(15, 60)
            out.append(
                Detection(
                    [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                    float(self.rng.uniform(0.12, 0.42)),
                    2,
                    "car",
                )
            )
        return out
