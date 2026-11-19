"""Metric motion estimation of tracked objects on the ground plane.

Each track gets its own constant-velocity Kalman filter fed with the fused
monocular range; the filter turns noisy per-frame positions into a smooth
position and a relative velocity, which time-to-collision needs.
"""

from __future__ import annotations

import numpy as np

from ..structures import Track, WorldState
from ..tracking.kalman import PointKalmanFilter


class GroundMotionEstimator:
    def __init__(self, accel_std: float = 3.0, max_idle: int = 30) -> None:
        self.accel_std = accel_std
        self.max_idle = max_idle
        self.filters: dict[int, PointKalmanFilter] = {}
        self.last_t: dict[int, float] = {}
        self.idle: dict[int, int] = {}

    def update(self, track: Track, x: float, z: float, sigma: float, t: float, x_sigma: float = 0.3) -> WorldState:
        tid = track.track_id
        # Lateral error: box-centre error plus range error propagated through the bearing
        sig = np.array([x_sigma + sigma * abs(x) / max(z, 1.0), max(0.1, sigma)])
        kf = self.filters.get(tid)
        if kf is None:
            kf = PointKalmanFilter(self.accel_std)
            kf.initiate(np.array([x, z]), sig)
            self.filters[tid] = kf
        else:
            dt = float(np.clip(t - self.last_t[tid], 1e-3, 1.0))
            kf.predict(dt)
            kf.update(np.array([x, z]), sig)
        self.last_t[tid] = t
        self.idle[tid] = 0
        m = kf.mean
        assert m is not None and kf.cov is not None
        return WorldState(
            x=float(m[0]),
            z=float(m[1]),
            vx=float(m[2]),
            vz=float(m[3]),
            distance_sigma=float(np.sqrt(kf.cov[1, 1])),
        )

    def prune(self, active_ids: set[int]) -> None:
        for tid in list(self.filters):
            if tid in active_ids:
                continue
            self.idle[tid] = self.idle.get(tid, 0) + 1
            if self.idle[tid] > self.max_idle:
                self.filters.pop(tid, None)
                self.last_t.pop(tid, None)
                self.idle.pop(tid, None)
