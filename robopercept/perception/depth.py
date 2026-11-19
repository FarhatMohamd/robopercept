"""Optional dense monocular depth (Depth Anything V2 / V3 through Hugging Face).

Relative models output an affine-invariant inverse depth. It is turned into
metres online by fitting ``1/z = a * d + b`` against the geometric ranges of the
tracked objects (robust least squares), so no metric fine-tuning is required.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..config import DepthConfig


class DepthEstimator:
    def __init__(self, cfg: DepthConfig) -> None:
        self.cfg = cfg
        self._pipe: Any = None
        self.scale: tuple[float, float] | None = None  # (a, b) for relative maps

    def _load(self) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("Dense depth needs `pip install robopercept[depth]`.") from exc
        self._pipe = pipeline(task="depth-estimation", model=self.cfg.model)

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Return a float32 map of the same size as ``image``."""
        if self._pipe is None:
            self._load()
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        out = self._pipe(pil)
        depth = out["predicted_depth"]
        arr = depth.squeeze().detach().cpu().numpy() if hasattr(depth, "detach") else np.asarray(depth)
        return cv2.resize(arr.astype(np.float32), (image.shape[1], image.shape[0]))

    @staticmethod
    def sample(depth_map: np.ndarray, box: np.ndarray) -> float:
        """Robust value inside the lower-central part of a box (avoids background)."""
        x1, y1, x2, y2 = box.astype(int)
        h, w = depth_map.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        xa, xb = max(0, x1 + bw // 4), min(w, x2 - bw // 4)
        ya, yb = max(0, y1 + bh // 2), min(h, y2 - bh // 10)
        if xb <= xa or yb <= ya:
            return float("nan")
        return float(np.median(depth_map[ya:yb, xa:xb]))

    def fit_scale(self, disparities: np.ndarray, ranges: np.ndarray) -> None:
        """Fit the relative-to-metric mapping with an iteratively reweighted LSQ."""
        d = np.asarray(disparities, dtype=np.float64)
        z = np.asarray(ranges, dtype=np.float64)
        ok = np.isfinite(d) & np.isfinite(z) & (z > 0.5)
        if ok.sum() < 3:
            return
        d, inv_z = d[ok], 1.0 / z[ok]
        A = np.stack([d, np.ones_like(d)], axis=1)
        w = np.ones_like(d)
        coef = np.zeros(2)
        for _ in range(5):
            coef, *_ = np.linalg.lstsq(A * w[:, None], inv_z * w, rcond=None)
            r = inv_z - A @ coef
            mad = np.median(np.abs(r)) + 1e-9
            w = 1.0 / np.maximum(1.0, np.abs(r) / (1.5 * mad))  # Huber weights
        if coef[0] > 0:
            prev = self.scale
            new = (float(coef[0]), float(coef[1]))
            # Smooth the scale over time to avoid jumps between frames
            self.scale = new if prev is None else tuple(0.7 * p + 0.3 * n for p, n in zip(prev, new))

    def to_metric(self, value: float) -> float | None:
        if not np.isfinite(value):
            return None
        if self.cfg.mode == "metric":
            return float(value)
        if self.scale is None:
            return None
        inv = self.scale[0] * value + self.scale[1]
        return float(1.0 / inv) if inv > 1e-6 else None
