"""Lightweight ego-lane detection (colour mask + edges + Hough + polynomial fit).

Deliberately classical: it runs in a few milliseconds on a CPU, needs no
weights and degrades gracefully (``valid = False``) when markings are absent,
which is the common case for indoor robots.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import LaneConfig
from ..structures import LaneEstimate


class LaneDetector:
    def __init__(self, cfg: LaneConfig | None = None) -> None:
        self.cfg = cfg or LaneConfig()
        self._left: np.ndarray | None = None
        self._right: np.ndarray | None = None
        self._miss = 0
        self._departing = False

    def _roi_mask(self, h: int, w: int) -> np.ndarray:
        top = int(h * self.cfg.roi_top_ratio)
        poly = np.array(
            [[(0.0, h), (0.42 * w, top), (0.58 * w, top), (w, h)]], dtype=np.int32
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, poly, 255)
        return mask

    @staticmethod
    def _marking_mask(image: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        white = cv2.inRange(hls, (0, 190, 0), (180, 255, 255))
        yellow = cv2.inRange(hls, (15, 90, 90), (35, 230, 255))
        return cv2.bitwise_or(white, yellow)

    def _fit_side(self, pts: list[tuple[float, float]], y_lo: int, y_hi: int) -> np.ndarray | None:
        if len(pts) < 6:
            return None
        arr = np.asarray(pts, dtype=np.float64)
        ys, xs = arr[:, 1], arr[:, 0]
        if ys.max() - ys.min() < 0.15 * (y_hi - y_lo):
            return None
        deg = 2 if len(pts) >= 12 else 1
        coef = np.polyfit(ys, xs, deg)
        resid = np.abs(np.polyval(coef, ys) - xs)
        inl = resid < max(4.0, 2.5 * np.median(resid))
        if inl.sum() >= 6:
            coef = np.polyfit(ys[inl], xs[inl], deg)
        if deg == 1:
            coef = np.r_[0.0, coef]
        return coef

    def detect(self, image: np.ndarray) -> LaneEstimate:
        h, w = image.shape[:2]
        y_lo, y_hi = int(h * self.cfg.roi_top_ratio), h - 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)
        mask = cv2.bitwise_and(edges, self._marking_mask(cv2.dilate(image, np.ones((3, 3)))))
        mask = cv2.bitwise_and(mask, self._roi_mask(h, w))
        lines = cv2.HoughLinesP(
            mask, 1, np.pi / 180, threshold=18, minLineLength=int(0.03 * h), maxLineGap=int(0.05 * h)
        )
        left_pts: list[tuple[float, float]] = []
        right_pts: list[tuple[float, float]] = []
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4).astype(float):
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.35:  # near-horizontal clutter (shadows, stop lines)
                    continue
                mid_x = (x1 + x2) / 2
                if slope < 0 and mid_x < w * 0.55:
                    left_pts += [(x1, y1), (x2, y2)]
                elif slope > 0 and mid_x > w * 0.45:
                    right_pts += [(x1, y1), (x2, y2)]

        left = self._fit_side(left_pts, y_lo, y_hi)
        right = self._fit_side(right_pts, y_lo, y_hi)
        a = self.cfg.smoothing
        if left is not None:
            self._left = left if self._left is None else a * self._left + (1 - a) * left
        if right is not None:
            self._right = right if self._right is None else a * self._right + (1 - a) * right
        if left is None and right is None:
            self._miss += 1
            if self._miss > 15:
                self._left = self._right = None
        else:
            self._miss = 0

        est = LaneEstimate(left=self._left, right=self._right, y_range=(y_lo, y_hi))
        if est.valid:
            xl = float(np.polyval(est.left, y_hi))
            xr = float(np.polyval(est.right, y_hi))
            width_px = xr - xl
            if width_px > 0.15 * w:
                centre = (xl + xr) / 2.0
                # Positive offset: the vehicle sits to the right of the lane centre
                est.offset_m = (w / 2.0 - centre) / width_px * self.cfg.lane_width_m
                # Hysteresis avoids flickering alarms around the threshold
                thr = self.cfg.departure_threshold_m
                limit = thr - 0.1 if self._departing else thr
                self._departing = abs(est.offset_m) > limit
                est.departure = self._departing
            else:
                est.left = est.right = None
        return est
