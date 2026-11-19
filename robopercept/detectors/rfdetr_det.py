"""RF-DETR backend (Roboflow, Apache-2.0 for N/S/M/Base sizes)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..structures import Detection
from .base import Detector

_SIZES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


class RFDETRDetector(Detector):
    name = "rfdetr"

    def __init__(
        self, model: str = "small", conf: float = 0.3, classes: list[str] | None = None
    ) -> None:
        super().__init__(conf, classes)
        try:
            import rfdetr
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("The 'rfdetr' backend needs `pip install robopercept[rfdetr]`.") from exc

        key = model.lower().replace("rf-detr-", "").replace("rfdetr-", "")
        cls_name = _SIZES.get(key)
        if cls_name is None:
            # Treat the argument as a fine-tuned checkpoint path
            self.model = rfdetr.RFDETRBase(pretrain_weights=model)
        else:
            self.model = getattr(rfdetr, cls_name)()
        try:
            from rfdetr.util.coco_classes import COCO_CLASSES

            self._names = dict(COCO_CLASSES)
        except ImportError:  # pragma: no cover
            self._names = {}
        if hasattr(self.model, "optimize_for_inference"):
            try:
                self.model.optimize_for_inference()
            except Exception:  # pragma: no cover - optional speed-up only
                pass

    @property
    def class_names(self) -> list[str]:
        return [self._names[k] for k in sorted(self._names)]

    def _predict(self, image: np.ndarray, meta: dict[str, Any] | None) -> list[Detection]:
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        det = self.model.predict(pil, threshold=self.conf)
        out: list[Detection] = []
        for box, score, cid in zip(det.xyxy, det.confidence, det.class_id):
            out.append(Detection(box, float(score), int(cid), str(self._names.get(int(cid), cid))))
        return out
