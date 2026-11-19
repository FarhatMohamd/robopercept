"""Pinhole camera with a known mounting height and pitch above a flat ground.

Ego frame: ``X`` to the right, ``Y`` downwards, ``Z`` forward, origin at the
camera centre projected on the vertical axis. The ground is the plane
``Y = height``. A positive pitch tilts the optical axis towards the ground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import CameraConfig


@dataclass
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    height: float = 1.4
    pitch: float = 0.0  # radians
    width: int = 0
    image_height: int = 0

    @classmethod
    def from_config(cls, cfg: CameraConfig, width: int, height: int) -> PinholeCamera:
        fx = cfg.fx
        if fx is None:
            fx = (width / 2.0) / math.tan(math.radians(cfg.hfov_deg) / 2.0)
        fy = cfg.fy if cfg.fy is not None else fx
        cx = cfg.cx if cfg.cx is not None else width / 2.0
        cy = cfg.cy if cfg.cy is not None else height / 2.0
        return cls(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            height=cfg.height_m,
            pitch=math.radians(cfg.pitch_deg),
            width=width,
            image_height=height,
        )

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    @property
    def horizon_v(self) -> float:
        """Image row of the horizon (vanishing line of the ground plane)."""
        return self.cy - self.fy * math.tan(self.pitch)

    def ego_to_cam(self, pts: np.ndarray) -> np.ndarray:
        """Rotate ego-frame points ``(N, 3)`` into the pitched camera frame."""
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        return np.stack([x, y * c - z * s, y * s + z * c], axis=1)

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project ego points to pixels. Returns ``(uv, valid)`` where ``valid`` means in front."""
        cam = self.ego_to_cam(pts)
        z = cam[:, 2]
        valid = z > 1e-3
        zs = np.where(valid, z, 1.0)
        u = self.fx * cam[:, 0] / zs + self.cx
        v = self.fy * cam[:, 1] / zs + self.cy
        return np.stack([u, v], axis=1), valid

    def project_ground(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project ground points given by lateral ``x`` and forward ``z`` coordinates."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        pts = np.stack([x, np.full_like(x, self.height), z], axis=1)
        return self.project(pts)

    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float] | None:
        """Back-project a pixel onto the ground plane. ``None`` above the horizon."""
        dx = (u - self.cx) / self.fx
        dy = (v - self.cy) / self.fy
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        ray_y = dy * c + s
        ray_z = -dy * s + c
        if ray_y <= 1e-6:
            return None
        t = self.height / ray_y
        return float(t * dx), float(t * ray_z)

    def ground_homography(self) -> np.ndarray:
        """3x3 homography mapping ground ``(x, z, 1)`` to image ``(u, v, 1)``."""
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        # Columns: image of the x axis, the z axis and the ground origin
        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([0.0, -s, c])
        t = np.array([0.0, self.height * c, self.height * s])
        return self.K @ np.stack([r1, r2, t], axis=1)
