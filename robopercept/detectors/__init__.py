"""Detector backends and a factory building one from the configuration."""

from __future__ import annotations

from ..config import DetectorConfig
from .base import Detector
from .coco import COCO_NAMES, ROAD_CLASSES
from .synthetic import SyntheticDetector

__all__ = ["COCO_NAMES", "ROAD_CLASSES", "Detector", "SyntheticDetector", "build_detector"]


def build_detector(cfg: DetectorConfig) -> Detector:
    """Instantiate the backend selected in ``cfg.backend``."""
    if cfg.backend == "synthetic":
        return SyntheticDetector(
            conf=cfg.conf,
            classes=cfg.classes,
            miss_rate=cfg.synthetic_miss_rate,
            box_noise_px=cfg.synthetic_box_noise_px,
            false_positive_rate=cfg.synthetic_false_positive_rate,
        )
    if cfg.backend == "onnx":
        from .onnx_det import OnnxDetector

        return OnnxDetector(
            cfg.model,
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            device=cfg.device,
            names=cfg.names or None,
            classes=cfg.classes,
            fmt=cfg.onnx_format,
            max_det=cfg.max_det,
        )
    if cfg.backend == "ultralytics":
        from .ultralytics_det import UltralyticsDetector

        return UltralyticsDetector(
            cfg.model,
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            device=cfg.device,
            half=cfg.half,
            max_det=cfg.max_det,
            classes=cfg.classes,
            prompts=cfg.prompts,
        )
    if cfg.backend == "rfdetr":
        from .rfdetr_det import RFDETRDetector

        return RFDETRDetector(cfg.model, conf=cfg.conf, classes=cfg.classes)
    raise ValueError(f"Unknown detector backend {cfg.backend!r}")
