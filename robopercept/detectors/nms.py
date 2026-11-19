"""Vectorised box utilities and non-maximum suppression in pure NumPy."""

from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[cx, cy, w, h]`` boxes to ``[x1, y1, x2, y2]``."""
    boxes = np.asarray(boxes, dtype=np.float32)
    out = np.empty_like(boxes)
    half_w = boxes[..., 2] / 2.0
    half_h = boxes[..., 3] / 2.0
    out[..., 0] = boxes[..., 0] - half_w
    out[..., 1] = boxes[..., 1] - half_h
    out[..., 2] = boxes[..., 0] + half_w
    out[..., 3] = boxes[..., 1] + half_h
    return out


def box_area(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes[..., 2] - boxes[..., 0], 0, None) * np.clip(
        boxes[..., 3] - boxes[..., 1], 0, None
    )


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``xyxy`` boxes, shape ``(len(a), len(b))``."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    """Greedy NMS. Returns the indices of kept boxes, highest score first."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    areas = box_area(boxes)
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_thresh: float = 0.5
) -> np.ndarray:
    """Class-aware NMS using the coordinate offset trick."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    offset = (np.asarray(class_ids, dtype=np.float32) * (float(boxes.max()) + 1.0))[:, None]
    return nms(boxes + offset, scores, iou_thresh)
