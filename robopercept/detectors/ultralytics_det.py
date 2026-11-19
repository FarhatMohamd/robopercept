"""Ultralytics backend: YOLO26, YOLO11, RT-DETR, YOLOE (open vocabulary) and YOLO-World.

PyTorch and ``ultralytics`` are imported lazily so that the rest of the package
works on machines where they are not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import Detection
from .base import Detector


class UltralyticsDetector(Detector):
    name = "ultralytics"

    def __init__(
        self,
        model: str = "yolo26n.pt",
        conf: float = 0.3,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str = "auto",
        half: bool = False,
        max_det: int = 300,
        classes: list[str] | None = None,
        prompts: list[str] | None = None,
    ) -> None:
        super().__init__(conf, classes)
        try:
            import ultralytics  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "The 'ultralytics' backend needs `pip install robopercept[yolo]`."
            ) from exc

        lowered = model.lower()
        if "rtdetr" in lowered:
            from ultralytics import RTDETR

            self.model = RTDETR(model)
        elif "yoloe" in lowered:
            from ultralytics import YOLOE

            self.model = YOLOE(model)
        elif "world" in lowered:
            from ultralytics import YOLOWorld

            self.model = YOLOWorld(model)
        else:
            from ultralytics import YOLO

            self.model = YOLO(model)

        if prompts:
            self._set_prompts(list(prompts))

        self.iou = iou
        self.imgsz = imgsz
        self.device = None if device == "auto" else device
        self.half = half
        self.max_det = max_det

    def _set_prompts(self, prompts: list[str]) -> None:
        """Configure open-vocabulary classes (YOLOE / YOLO-World)."""
        model = self.model
        if not hasattr(model, "set_classes"):
            raise ValueError("Text prompts require a YOLOE or YOLO-World model")
        try:
            model.set_classes(prompts, model.get_text_pe(prompts))  # YOLOE API
        except (AttributeError, TypeError):
            model.set_classes(prompts)  # YOLO-World / newer YOLOE API

    @property
    def class_names(self) -> list[str]:
        names = getattr(self.model, "names", {}) or {}
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
        return [str(n) for n in names]

    def _predict(self, image: np.ndarray, meta: dict[str, Any] | None) -> list[Detection]:
        results = self.model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            max_det=self.max_det,
            verbose=False,
        )
        if not results:
            return []
        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        names = res.names or {}
        return [
            Detection(b, float(s), int(c), str(names.get(int(c), c)))
            for b, s, c in zip(xyxy, scores, cls)
        ]
