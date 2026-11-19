"""Global camera motion compensation from sparse optical flow.

When the robot or vehicle moves, every object shifts in the image even if it is
static. Estimating that global motion and applying it to the track predictions
keeps identities stable on a moving platform (idea popularised by BoT-SORT).
"""

from __future__ import annotations

import cv2
import numpy as np


class CameraMotionCompensator:
    def __init__(self, downscale: int = 2, max_corners: int = 400) -> None:
        self.downscale = max(1, int(downscale))
        self.max_corners = max_corners
        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_pts = None

    def estimate(self, image: np.ndarray, mask_boxes: np.ndarray | None = None) -> np.ndarray:
        """Return the 2x3 affine warp mapping the previous frame onto the current one."""
        warp = np.eye(2, 3, dtype=np.float64)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if self.downscale > 1:
            gray = cv2.resize(gray, None, fx=1 / self.downscale, fy=1 / self.downscale)

        # Ignore moving objects: features are taken from the background only
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        if mask_boxes is not None:
            for x1, y1, x2, y2 in (np.asarray(mask_boxes) / self.downscale).astype(int):
                mask[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = 0

        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=8, mask=mask
        )
        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) >= 8:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
            good = status.reshape(-1) == 1
            if good.sum() >= 8:
                m, _ = cv2.estimateAffinePartial2D(
                    self.prev_pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=2.0
                )
                if m is not None:
                    warp = m.astype(np.float64)
                    warp[:, 2] *= self.downscale
        self.prev_gray = gray
        self.prev_pts = pts
        return warp


def apply_warp_to_state(mean: np.ndarray, cov: np.ndarray, warp: np.ndarray) -> None:
    """Apply an affine warp in place to a ``(cx, cy, a, h, vx, vy, va, vh)`` state."""
    R = warp[:, :2]
    t = warp[:, 2]
    scale = float(np.sqrt(abs(np.linalg.det(R))))
    mean[:2] = R @ mean[:2] + t
    mean[4:6] = R @ mean[4:6]
    mean[3] *= scale
    mean[7] *= scale
    T = np.eye(8)
    T[:2, :2] = R
    T[4:6, 4:6] = R
    cov[:] = T @ cov @ T.T
