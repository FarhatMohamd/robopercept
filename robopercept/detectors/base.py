"""Detector interface. Every backend turns an image into a list of detections."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..types import Detection


class Detector(ABC):
    """Abstract object detector.

    Implementations must be cheap to call repeatedly; heavy initialisation
    (weights loading, graph compilation) belongs in ``__init__`` or ``warmup``.
    """

    name: str = "detector"

    def __init__(self, conf: float = 0.3, classes: list[str] | None = None) -> None:
        self.conf = float(conf)
        self.keep_classes = {c.lower() for c in (classes or [])}

    @property
    def class_names(self) -> list[str]:
        return []

    @abstractmethod
    def _predict(self, image: np.ndarray, meta: dict[str, Any] | None) -> list[Detection]:
        """Backend specific inference on a BGR ``uint8`` image."""

    def detect(self, image: np.ndarray, meta: dict[str, Any] | None = None) -> list[Detection]:
        dets = self._predict(image, meta)
        dets = [d for d in dets if d.score >= self.conf]
        if self.keep_classes:
            dets = [d for d in dets if d.class_name.lower() in self.keep_classes]
        return dets

    def warmup(self, shape: tuple[int, int, int] = (640, 640, 3)) -> None:
        self._predict(np.zeros(shape, dtype=np.uint8), None)

    def close(self) -> None:  # pragma: no cover - optional hook
        pass
