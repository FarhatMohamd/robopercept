"""Typed configuration loaded from YAML files.

Every field has a sensible default so that ``Config()`` alone gives a working
pipeline on the built-in synthetic scene. Values can be overridden from the
command line with dotted keys, e.g. ``--set detector.conf=0.4``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceConfig(_Section):
    # Camera index ("0"), video file, RTSP/HTTP URL, image folder or "synthetic"
    uri: str = "synthetic"
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    loop: bool = False
    # Always deliver the newest frame and drop stale ones (live sources)
    drop_frames: bool = True
    max_frames: int | None = None


class DetectorConfig(_Section):
    backend: Literal["synthetic", "ultralytics", "onnx", "rfdetr"] = "synthetic"
    # Ultralytics weights (yolo26n.pt, yolo11s.pt, rtdetr-l.pt, yoloe-26s-seg.pt) or ONNX path
    model: str = "yolo26n.pt"
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.30
    iou: float = 0.50
    half: bool = False
    max_det: int = 300
    # Keep only these class names (empty list keeps every class)
    classes: list[str] = Field(default_factory=list)
    # Open-vocabulary text prompts (YOLOE / YOLO-World backends)
    prompts: list[str] = Field(default_factory=list)
    # Class names for ONNX models exported without metadata
    names: list[str] = Field(default_factory=list)
    # ONNX output layout: "auto", "end2end" (YOLO26/YOLOv10), "yolo" (v8/11 + NMS), "detr"
    onnx_format: Literal["auto", "end2end", "yolo", "detr"] = "auto"
    # Synthetic backend: imperfections injected on top of ground truth
    synthetic_miss_rate: float = 0.04
    synthetic_box_noise_px: float = 1.5
    synthetic_false_positive_rate: float = 0.02


class TrackerConfig(_Section):
    enabled: bool = True
    high_thresh: float = 0.50
    low_thresh: float = 0.10
    new_track_thresh: float = 0.55
    match_thresh: float = 0.80
    second_match_thresh: float = 0.50
    track_buffer: int = 30
    min_hits: int = 2
    # Sparse optical-flow camera motion compensation (moving robot / vehicle)
    camera_motion_compensation: bool = False
    history: int = 40


class CameraConfig(_Section):
    # Pinhole intrinsics. ``None`` means "derive from horizontal field of view".
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    hfov_deg: float = 70.0
    # Mounting height above the ground (m) and downward pitch (deg)
    height_m: float = 1.40
    pitch_deg: float = 0.0


class DistanceConfig(_Section):
    enabled: bool = True
    # Real-world object heights (m) used by the size prior
    class_heights: dict[str, float] = Field(
        default_factory=lambda: {
            "person": 1.70,
            "pedestrian": 1.70,
            "bicycle": 1.10,
            "cyclist": 1.70,
            "motorcycle": 1.30,
            "car": 1.50,
            "truck": 3.20,
            "bus": 3.10,
            "traffic light": 0.90,
            "stop sign": 0.75,
            "traffic cone": 0.70,
            "forklift": 2.10,
            "pallet": 0.15,
            "dog": 0.60,
        }
    )
    # Relative standard deviation of each ranging cue
    ground_sigma_rel: float = 0.06
    size_sigma_rel: float = 0.15
    max_range_m: float = 120.0


class DepthConfig(_Section):
    # Optional dense monocular depth (Depth Anything V2 through transformers)
    enabled: bool = False
    model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    every_n_frames: int = 3
    # "relative" maps are rescaled online against geometric ranging
    mode: Literal["relative", "metric"] = "relative"


class LaneConfig(_Section):
    enabled: bool = True
    roi_top_ratio: float = 0.58
    lane_width_m: float = 3.5
    departure_threshold_m: float = 0.55
    smoothing: float = 0.6


class ZoneConfig(_Section):
    name: str
    # Polygon vertices, either normalized image coords [0..1] or ground metres
    polygon: list[tuple[float, float]]
    frame: Literal["image", "ground"] = "image"
    classes: list[str] = Field(default_factory=list)
    level: Literal["caution", "warning", "critical"] = "warning"
    dwell_s: float = 0.0


class SafetyConfig(_Section):
    profile: Literal["vehicle", "robot"] = "vehicle"
    # Ego corridor half-width (m): vehicle half width + margin
    corridor_half_width_m: float = 1.30
    ttc_warning_s: float = 3.0
    ttc_critical_s: float = 1.6
    # Speed & separation monitoring (robot profile)
    stop_distance_m: float = 1.0
    slow_distance_m: float = 3.0
    ego_speed_mps: float = 0.0
    reaction_time_s: float = 0.8
    max_decel_mps2: float = 6.0
    vulnerable_classes: list[str] = Field(
        default_factory=lambda: ["person", "pedestrian", "bicycle", "cyclist", "motorcycle", "dog"]
    )
    # Frames a risk must persist before being raised (debounce)
    persistence_frames: int = 2


class OutputConfig(_Section):
    show: bool = False
    save_video: str | None = None
    jsonl_log: str | None = None
    event_snapshots_dir: str | None = None
    draw_bev: bool = True
    draw_trails: bool = True


class ServerConfig(_Section):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    jpeg_quality: int = 80


class MqttConfig(_Section):
    enabled: bool = False
    host: str = "localhost"
    port: int = 1883
    topic: str = "robopercept"


class Config(_Section):
    source: SourceConfig = Field(default_factory=SourceConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    distance: DistanceConfig = Field(default_factory=DistanceConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    lanes: LaneConfig = Field(default_factory=LaneConfig)
    zones: list[ZoneConfig] = Field(default_factory=list)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    mqtt: MqttConfig = Field(default_factory=MqttConfig)


def _parse_scalar(raw: str) -> Any:
    """Interpret a command-line override value with YAML semantics."""
    return yaml.safe_load(raw)


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key.sub=value`` overrides to a nested dictionary in place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = data
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override {key!r}: {part!r} is not a section")
        node[parts[-1]] = _parse_scalar(raw)
    return data


PRESETS_DIR = Path(__file__).resolve().parent / "presets"


def resolve_config_path(path: str | Path) -> Path:
    """Accept a file path or the name of a bundled preset (``urban``, ``warehouse``)."""
    p = Path(path)
    if p.exists():
        return p
    preset = PRESETS_DIR / f"{p.stem}.yaml"
    if preset.exists():
        return preset
    raise FileNotFoundError(f"Configuration not found: {path} (presets: {list_presets()})")


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML file (or bundled preset) and apply optional dotted overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        path = resolve_config_path(path)
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    if overrides:
        apply_overrides(data, overrides)
    return Config.model_validate(data)


def dump_config(cfg: Config) -> str:
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
