import numpy as np

from robopercept.detectors.nms import batched_nms, iou_matrix, nms, xywh_to_xyxy
from robopercept.detectors.onnx_det import decode_output, detect_format, letterbox, scale_boxes


def test_iou_matrix_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 50 / 150, 0.0], atol=1e-6)
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_nms_keeps_best_and_separates_classes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]
    # Same overlapping boxes but different classes survive class-aware NMS
    keep = batched_nms(boxes, scores, np.array([0, 1, 0]), 0.5)
    assert sorted(keep.tolist()) == [0, 1, 2]


def test_xywh_conversion():
    np.testing.assert_allclose(xywh_to_xyxy(np.array([[10, 20, 4, 6]])), [[8, 17, 12, 23]])


def test_decode_end2end_layout():
    out = np.zeros((1, 300, 6), dtype=np.float32)
    out[0, 0] = [10, 20, 110, 220, 0.9, 2]
    out[0, 1] = [0, 0, 5, 5, 0.1, 0]
    assert detect_format(out[0]) == "end2end"
    boxes, scores, cls = decode_output(out, "auto", 0.25, 0.5, (640, 640))
    assert len(boxes) == 1 and cls[0] == 2
    np.testing.assert_allclose(boxes[0], [10, 20, 110, 220])


def test_decode_yolo_raw_layout_applies_nms():
    n_cls, anchors = 3, 50
    out = np.zeros((1, 4 + n_cls, anchors), dtype=np.float32)
    # two overlapping candidates of class 1, one isolated of class 2
    out[0, :4, 0] = [100, 100, 40, 40]
    out[0, 4 + 1, 0] = 0.9
    out[0, :4, 1] = [102, 101, 40, 40]
    out[0, 4 + 1, 1] = 0.8
    out[0, :4, 2] = [400, 300, 60, 30]
    out[0, 4 + 2, 2] = 0.7
    assert detect_format(out[0]) == "yolo"
    boxes, scores, cls = decode_output(out, "auto", 0.25, 0.5, (640, 640))
    assert cls.tolist() == [1, 2]
    np.testing.assert_allclose(boxes[0], [80, 80, 120, 120])


def test_decode_detr_layout_normalised_boxes():
    out = np.zeros((1, 300, 4 + 2), dtype=np.float32)
    out[0, :, :4] = 0.1
    out[0, :, 4:] = -10.0  # logits
    out[0, 5] = [0.5, 0.5, 0.25, 0.5, -10.0, 5.0]
    boxes, scores, cls = decode_output(out, "detr", 0.5, 0.5, (640, 640))
    assert cls.tolist() == [1]
    np.testing.assert_allclose(boxes[0], [240, 160, 400, 480], atol=1e-4)


def test_letterbox_roundtrip():
    img = np.zeros((480, 960, 3), dtype=np.uint8)
    padded, scale, pad = letterbox(img, 640)
    assert padded.shape == (640, 640, 3)
    assert scale == 640 / 960
    box_in_input = np.array([[pad[0] + 64, pad[1] + 32, pad[0] + 128, pad[1] + 96]], dtype=np.float32)
    np.testing.assert_allclose(scale_boxes(box_in_input, scale, pad, (480, 960))[0], [96, 48, 192, 144], atol=1e-4)
