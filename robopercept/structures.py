"""Core data structures shared by every stage of the perception pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class RiskLevel(IntEnum):
    """Ordered risk levels: comparisons such as ``level >= RiskLevel.WARNING`` are valid."""

    SAFE = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass
class Detection:
    """A single object detection in image coordinates.

    ``box`` is ``[x1, y1, x2, y2]`` in pixels.
    """

    box: np.ndarray
    score: float
    class_id: int
    class_name: str = ""
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.box = np.asarray(self.box, dtype=np.float32).reshape(4)

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Ground contact point, the reference used for monocular ranging."""
        return float((self.box[0] + self.box[2]) / 2.0), float(self.box[3])


@dataclass
class WorldState:
    """Metric state of an object in the ego frame.

    Ego frame convention (right-handed, ground plane):
      * ``x`` lateral, positive to the right of the camera (m)
      * ``z`` longitudinal, positive forward (m)
    """

    x: float
    z: float
    vx: float = 0.0
    vz: float = 0.0
    distance_sigma: float = float("nan")

    @property
    def distance(self) -> float:
        return float(np.hypot(self.x, self.z))

    @property
    def closing_speed(self) -> float:
        """Positive when the object approaches the ego origin (m/s)."""
        d = self.distance
        if d < 1e-6:
            return 0.0
        return float(-(self.x * self.vx + self.z * self.vz) / d)


@dataclass
class Track:
    """A tracked object with a persistent identity."""

    track_id: int
    box: np.ndarray
    score: float
    class_id: int
    class_name: str = ""
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    world: WorldState | None = None
    ttc: float = float("inf")
    risk: RiskLevel = RiskLevel.SAFE
    in_ego_path: bool = False
    zones: list[str] = field(default_factory=list)
    history: list[tuple[float, float]] = field(default_factory=list)
    # Raw box of the detection matched on this frame (``box`` is the filtered one)
    det_box: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": int(self.track_id),
            "box": [round(float(v), 1) for v in self.box],
            "score": round(float(self.score), 3),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "risk": self.risk.label,
            "in_ego_path": bool(self.in_ego_path),
            "zones": list(self.zones),
        }
        if np.isfinite(self.ttc):
            out["ttc"] = round(float(self.ttc), 2)
        if self.world is not None:
            out["world"] = {
                "x": round(self.world.x, 2),
                "z": round(self.world.z, 2),
                "vx": round(self.world.vx, 2),
                "vz": round(self.world.vz, 2),
                "distance": round(self.world.distance, 2),
            }
        return out


@dataclass
class LaneEstimate:
    """Ego-lane boundaries as image-space polynomials ``x = a*y^2 + b*y + c``."""

    left: np.ndarray | None = None
    right: np.ndarray | None = None
    y_range: tuple[int, int] = (0, 0)
    offset_m: float | None = None
    departure: bool = False

    @property
    def valid(self) -> bool:
        return self.left is not None and self.right is not None


@dataclass
class SafetyDecision:
    """Output of the safety layer, consumed by a motion controller."""

    level: RiskLevel = RiskLevel.SAFE
    speed_factor: float = 1.0
    reason: str = "clear"
    critical_track: int | None = None
    min_ttc: float = float("inf")
    min_distance: float = float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.label,
            "speed_factor": round(float(self.speed_factor), 3),
            "reason": self.reason,
            "critical_track": self.critical_track,
            "min_ttc": None if not np.isfinite(self.min_ttc) else round(float(self.min_ttc), 2),
            "min_distance": None if not np.isfinite(self.min_distance) else round(float(self.min_distance), 2),
        }


@dataclass
class Event:
    """A discrete event emitted by the pipeline (zone entry, collision warning, ...)."""

    kind: str
    frame_index: int
    timestamp: float
    message: str
    track_id: int | None = None
    level: RiskLevel = RiskLevel.CAUTION
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "frame": self.frame_index,
            "t": round(self.timestamp, 3),
            "message": self.message,
            "track_id": self.track_id,
            "level": self.level.label,
            "data": self.data,
        }


@dataclass
class FrameResult:
    """Everything the pipeline knows about one processed frame."""

    frame_index: int
    timestamp: float
    image: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    lanes: LaneEstimate | None = None
    decision: SafetyDecision = field(default_factory=SafetyDecision)
    events: list[Event] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    depth_map: np.ndarray | None = None
    # Source metadata (ground truth for the simulator, odometry, file path...)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame_index,
            "t": round(self.timestamp, 3),
            "n_detections": len(self.detections),
            "tracks": [t.to_dict() for t in self.tracks],
            "decision": self.decision.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "lanes": None
            if self.lanes is None
            else {
                "valid": self.lanes.valid,
                "offset_m": None if self.lanes.offset_m is None else round(float(self.lanes.offset_m), 2),
                "departure": bool(self.lanes.departure),
            },
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
        }
