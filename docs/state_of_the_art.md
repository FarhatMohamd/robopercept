# State of the art and design choices (November 2026)

This note explains why RoboPercept is built the way it is, and where each
component sits with respect to current methods.

## Real-time detectors

| Family | Representative models | What matters for robots |
|--------|----------------------|-------------------------|
| End-to-end CNN | **YOLO26** (Ultralytics, January 2026), YOLOv10 | No NMS at inference: constant latency in crowded scenes and simpler export to TensorRT / OpenVINO / ONNX. YOLO26 also drops the DFL head, adds ProgLoss and small-target-aware label assignment (STAL) and trains with the MuSGD optimiser. The YOLO26 paper reports 40.9–57.5 mAP on COCO at 1.7–11.8 ms on a T4 with TensorRT FP16. |
| Real-time DETR | RT-DETR, D-FINE, DEIM / DEIMv2, **RT-DETRv4**, **RF-DETR** | Transformer detectors are NMS-free by design. RT-DETRv4-L reports 55.4 AP at 124 FPS; RF-DETR (ICLR 2026) is the first real-time detector above 60 AP on COCO and was designed for transfer to new domains (RF100-VL benchmark). |
| Open vocabulary | **YOLOE-26**, YOLO-World | Classes given as text or visual prompts at run time: useful for warehouses and labs whose objects are not in COCO (pallets, carts, signs). |

**Choice.** A backend interface rather than one model: YOLO26 through
Ultralytics as the default, RT-DETR / RF-DETR for transformer detectors, any
ONNX export for PyTorch-free targets, YOLOE-26 for open vocabulary. The ONNX
decoder handles the three output layouts in use today (end-to-end, raw YOLO
anchors, DETR queries).

## Multi-object tracking

Tracking-by-detection dominates online robotics: SORT, DeepSORT, ByteTrack,
OC-SORT, BoT-SORT and recent variants. ByteTrack's key idea is to associate
low-confidence boxes in a second pass instead of discarding them, which
recovers partially occluded objects. OC-SORT improves non-linear motion;
BoT-SORT adds camera motion compensation and appearance re-identification.

**Choice.** ByteTrack (online, no appearance model, CPU-cheap) plus
BoT-SORT-style camera motion compensation for moving platforms, and
score-weighted class voting against label flicker. Re-identification is on
the roadmap for long occlusions.

## Monocular depth

Foundation models such as **Depth Anything V2** (2024) and **Depth Anything 3**
(November 2025) give dense, sharp depth from one image; metric variants exist
(DA2 metric outdoor/indoor fine-tunes, DA3-Metric, for which metric depth is
`focal × output / 300`). Independent studies evaluate DA2 as a low-cost
complement to LiDAR on robots. Their cost (tens of milliseconds on a GPU) and
the scale ambiguity of relative models make them a complement rather than a
replacement for geometry.

**Choice.** Geometric ranging (ground contact and size prior) always on,
fused by inverse variance; dense depth optional, every N frames, with online
scale fitting against the geometric ranges.

## Robotics integration

ROS 2 Jazzy is the current LTS distribution; `vision_msgs` 4.x with
`Detection2DArray` is the de-facto exchange format for 2-D detections and is
consumed by downstream fusion and navigation packages.

**Choice.** Publish `Detection2DArray` with track ids and metric poses, RViz
markers, and a speed factor applied by a separate fail-safe `cmd_vel` governor
so that the planner stays unchanged.

## References

* Ultralytics, *YOLO26: Unified Real-Time End-to-End Vision Models*, arXiv:2606.03748, 2026. https://docs.ultralytics.com/models/yolo26/
* R. Sapkota, M. Karkee, *YOLOE-26: Integrating YOLO26 with YOLOE for Real-Time Open-Vocabulary Instance Segmentation*, arXiv:2602.00168, 2026.
* I. Robinson et al., *RF-DETR: Neural Architecture Search for Real-Time Detection Transformers*, ICLR 2026, arXiv:2511.09554.
* *RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models*, arXiv:2510.25257, 2025.
* S. Huang et al., *Real-Time Object Detection Meets DINOv3* (DEIMv2), arXiv:2509.20787, 2025.
* Y. Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022.
* J. Cao et al., *Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking*, CVPR 2023, arXiv:2203.14360.
* N. Aharon et al., *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*, arXiv:2206.14651, 2022.
* L. Yang et al., *Depth Anything V2*, NeurIPS 2024.
* H. Lin et al., *Depth Anything 3: Recovering the Visual Space from Any Views*, arXiv:2511.10647, 2025.
* K. Bernardin, R. Stiefelhagen, *Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics*, EURASIP JIVP, 2008.
* E. Ristani et al., *Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking* (IDF1), ECCV Workshops 2016.
* ROS perception, `vision_msgs` 4.2. https://github.com/ros-perception/vision_msgs
