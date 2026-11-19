"""Pure-Python helpers converting RoboPercept results to ROS conventions.

Kept free of ROS imports so they can be unit-tested without a ROS install.
RoboPercept's ego frame is (x right, z forward); ROS REP-103 base frames are
(x forward, y left, z up).
"""

from __future__ import annotations


def ego_to_ros(x_right: float, z_forward: float) -> tuple[float, float]:
    """Map an ego-frame ground point to REP-103 ``(x forward, y left)``."""
    return z_forward, -x_right


def risk_color(level: str) -> tuple[float, float, float, float]:
    return {
        "safe": (0.30, 0.76, 0.54, 0.9),
        "caution": (0.94, 0.77, 0.24, 0.9),
        "warning": (1.00, 0.60, 0.24, 0.95),
        "critical": (0.94, 0.32, 0.31, 1.0),
    }.get(level, (0.6, 0.6, 0.6, 0.8))


def scale_twist(linear: tuple[float, float, float], angular: tuple[float, float, float], factor: float):
    """Scale a velocity command; only forward motion is limited (reversing stays allowed)."""
    factor = min(max(factor, 0.0), 1.0)
    lx, ly, lz = linear
    if lx > 0.0:
        lx *= factor
    return (lx, ly * factor, lz), tuple(a * max(factor, 0.3) for a in angular)
