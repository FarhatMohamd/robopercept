"""ONNX Runtime backend: runs exported models without PyTorch.

Supported output layouts (auto-detected):

* ``end2end``: ``(1, N, 6)`` rows ``[x1, y1, x2, y2, score, class]`` as produced by
  NMS-free models such as YOLO26 or YOLOv10.
* ``yolo``: ``(1, 4 + C, A)`` raw anchor predictions ``[cx, cy, w, h, scores...]``
  as produced by YOLOv8 / YOLO11; NMS is applied here.
* ``detr``: ``(1, Q, 4 + C)`` with normalised ``[cx, cy, w, h]`` boxes and class
  scores, as produced by RT-DETR style exports.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..structures import Detection
from .base import Detector
from .coco import COCO_NAMES
from .nms import batched_nms, xywh_to_xyxy


def letterbox(
    image: np.ndarray, size: int | tuple[int, int], color: int = 114
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize with unchanged aspect ratio and pad to ``size``.

    Returns the padded image, the scale factor and the ``(pad_x, pad_y)`` offsets.
    """
    th, tw = (size, size) if isinstance(size, int) else size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (tw - nw) / 2.0, (th - nh) / 2.0
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    out = np.full((th, tw, 3), color, dtype=np.uint8)
    out[top : top + nh, left : left + nw] = resized
    return out, scale, (float(left), float(top))


def decode_output(
    output: np.ndarray,
    fmt: str,
    conf: float,
    iou: float,
    input_size: tuple[int, int],
    max_det: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a raw model output into ``(boxes_xyxy, scores, class_ids)`` in input pixels."""
    out = np.asarray(output, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    if fmt == "auto":
        fmt = detect_format(out)

    if fmt == "end2end":
        boxes, scores, cls = out[:, :4], out[:, 4], out[:, 5].astype(np.int64)
        keep = scores >= conf
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    elif fmt == "yolo":
        if out.shape[0] < out.shape[1]:
            out = out.T  # (A, 4 + C)
        cls_scores = out[:, 4:]
        cls = cls_scores.argmax(axis=1)
        scores = cls_scores[np.arange(len(cls)), cls]
        keep = scores >= conf
        boxes = xywh_to_xyxy(out[keep, :4])
        scores, cls = scores[keep], cls[keep]
        idx = batched_nms(boxes, scores, cls, iou)
        boxes, scores, cls = boxes[idx], scores[idx], cls[idx]
    elif fmt == "detr":
        logits = out[:, 4:]
        # Accept both probabilities and logits
        probs = logits if (logits.min() >= 0 and logits.max() <= 1) else 1 / (1 + np.exp(-logits))
        cls = probs.argmax(axis=1)
        scores = probs[np.arange(len(cls)), cls]
        keep = scores >= conf
        th, tw = input_size
        boxes = xywh_to_xyxy(out[keep, :4]) * np.array([tw, th, tw, th], dtype=np.float32)
        scores, cls = scores[keep], cls[keep]
    else:
        raise ValueError(f"Unknown ONNX output format {fmt!r}")

    order = np.argsort(-scores)[:max_det]
    return boxes[order], scores[order], cls[order].astype(np.int64)


def detect_format(out: np.ndarray) -> str:
    """Guess the output layout of a 2-D prediction tensor."""
    rows, cols = out.shape
    if cols == 6:
        return "end2end"
    if rows < cols:
        return "yolo"  # (4 + C, anchors) is much wider than tall
    if np.all(out[:, :4] <= 1.5) and np.all(out[:, :4] >= -0.5):
        return "detr"
    return "yolo"


def scale_boxes(
    boxes: np.ndarray, scale: float, pad: tuple[float, float], shape: tuple[int, int]
) -> np.ndarray:
    """Map boxes from letterboxed input space back to the original image."""
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
    h, w = shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


class OnnxDetector(Detector):
    name = "onnx"

    def __init__(
        self,
        model: str,
        conf: float = 0.3,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str = "auto",
        names: list[str] | None = None,
        classes: list[str] | None = None,
        fmt: str = "auto",
        max_det: int = 300,
    ) -> None:
        super().__init__(conf, classes)
        import onnxruntime as ort

        if not Path(model).exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model}. Export one with "
                "`python scripts/export_model.py --model yolo26n.pt --format onnx`."
            )
        available = ort.get_available_providers()
        wanted = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cpu":
            wanted = ["CPUExecutionProvider"]
        providers = [p for p in wanted if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        h = shape[2] if isinstance(shape[2], int) else imgsz
        w = shape[3] if isinstance(shape[3], int) else imgsz
        self.input_size = (int(h), int(w))
        self.iou = iou
        self.fmt = fmt
        self.max_det = max_det
        self._names = names or self._names_from_metadata() or list(COCO_NAMES)

    def _names_from_metadata(self) -> list[str]:
        meta = self.session.get_modelmeta().custom_metadata_map
        raw = meta.get("names")
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, dict):
            return [str(parsed[k]) for k in sorted(parsed)]
        return [str(v) for v in parsed]

    @property
    def class_names(self) -> list[str]:
        return self._names

    def _predict(self, image: np.ndarray, meta: dict[str, Any] | None) -> list[Detection]:
        padded, scale, pad = letterbox(image, self.input_size)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        output = self.session.run(None, {self.input_name: blob})[0]
        boxes, scores, cls = decode_output(
            output, self.fmt, self.conf, self.iou, self.input_size, self.max_det
        )
        boxes = scale_boxes(boxes, scale, pad, image.shape[:2])
        names = self._names
        return [
            Detection(b, float(s), int(c), names[c] if c < len(names) else str(int(c)))
            for b, s, c in zip(boxes, scores, cls)
        ]
