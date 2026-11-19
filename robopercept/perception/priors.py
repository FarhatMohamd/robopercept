"""Typical physical dimensions of common road and indoor objects (metres).

``width`` is measured across the direction of travel, ``length`` along it.
Heights live in :class:`~robopercept.config.DistanceConfig` because they drive
the size cue and are worth tuning per deployment.
"""

from __future__ import annotations

# class name -> (width, length)
FOOTPRINT: dict[str, tuple[float, float]] = {
    "person": (0.5, 0.4),
    "pedestrian": (0.5, 0.4),
    "dog": (0.3, 0.8),
    "bicycle": (0.6, 1.8),
    "cyclist": (0.6, 1.8),
    "motorcycle": (0.8, 2.1),
    "car": (1.8, 4.4),
    "truck": (2.5, 7.5),
    "bus": (2.55, 11.0),
    "forklift": (1.2, 2.4),
    "pallet": (1.2, 0.8),
    "traffic cone": (0.4, 0.4),
}

DEFAULT_FOOTPRINT = (1.0, 1.0)


def footprint(class_name: str) -> tuple[float, float]:
    return FOOTPRINT.get(class_name.lower(), DEFAULT_FOOTPRINT)


def half_width(class_name: str) -> float:
    return footprint(class_name)[0] / 2.0
