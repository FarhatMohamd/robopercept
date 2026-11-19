"""Constant-velocity Kalman filter in ``(cx, cy, aspect, height)`` box space.

This is the motion model used by SORT/ByteTrack: the measurement is a bounding
box and the noise is scaled with the box height so that the filter behaves the
same for near (large) and far (small) objects.
"""

from __future__ import annotations

import numpy as np

# 95 % quantiles of the chi-square distribution, used for gating
CHI2INV95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}


class BoxKalmanFilter:
    ndim = 4

    def __init__(self, dt: float = 1.0) -> None:
        n = self.ndim
        self.F = np.eye(2 * n)
        self.F[:n, n:] = dt * np.eye(n)
        self.H = np.eye(n, 2 * n)
        self.std_pos = 1.0 / 20
        self.std_vel = 1.0 / 160

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[z, np.zeros(self.ndim)]
        h = z[3]
        std = [
            2 * self.std_pos * h,
            2 * self.std_pos * h,
            1e-2,
            2 * self.std_pos * h,
            10 * self.std_vel * h,
            10 * self.std_vel * h,
            1e-5,
            10 * self.std_vel * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [
            self.std_pos * h,
            self.std_pos * h,
            1e-2,
            self.std_pos * h,
            self.std_vel * h,
            self.std_vel * h,
            1e-5,
            self.std_vel * h,
        ]
        Q = np.diag(np.square(std))
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + Q
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        R = np.diag(np.square([self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]))
        return self.H @ mean, self.H @ cov @ self.H.T + R

    def update(
        self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, cov)
        # Solve instead of inverting for numerical stability
        gain = np.linalg.solve(proj_cov, (cov @ self.H.T).T).T
        mean = mean + gain @ (z - proj_mean)
        cov = cov - gain @ proj_cov @ gain.T
        return mean, cov

    def gating_distance(self, mean: np.ndarray, cov: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance between the state and each measurement."""
        proj_mean, proj_cov = self.project(mean, cov)
        d = np.atleast_2d(zs) - proj_mean
        L = np.linalg.cholesky(proj_cov)
        y = np.linalg.solve(L, d.T)
        return np.sum(y * y, axis=0)


class PointKalmanFilter:
    """2-D constant-velocity filter ``[x, z, vx, vz]`` for metric ground positions.

    Unlike the box filter it accepts a variable time step and a per-measurement
    noise, which is what monocular ranging provides (noise grows with distance).
    """

    def __init__(self, accel_std: float = 2.5) -> None:
        self.accel_std = accel_std
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None

    def initiate(self, xz: np.ndarray, sigma: np.ndarray) -> None:
        self.mean = np.r_[np.asarray(xz, dtype=float), 0.0, 0.0]
        self.cov = np.diag(np.r_[np.square(sigma), 4.0, 4.0])

    def predict(self, dt: float) -> None:
        assert self.mean is not None and self.cov is not None
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.accel_std**2
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        self.mean = F @ self.mean
        self.cov = F @ self.cov @ F.T + G @ (q * np.eye(2)) @ G.T

    def update(self, xz: np.ndarray, sigma: np.ndarray) -> None:
        assert self.mean is not None and self.cov is not None
        H = np.eye(2, 4)
        R = np.diag(np.square(sigma))
        S = H @ self.cov @ H.T + R
        K = np.linalg.solve(S, (self.cov @ H.T).T).T
        self.mean = self.mean + K @ (np.asarray(xz, dtype=float) - H @ self.mean)
        self.cov = (np.eye(4) - K @ H) @ self.cov
