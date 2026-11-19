# Changelog

## 1.0.0 — 2026-11-19

First public release.

* Detector backends: Ultralytics (YOLO26, YOLO11, RT-DETR, YOLOE-26, YOLO-World), ONNX Runtime with automatic output-layout detection, RF-DETR, simulator.
* ByteTrack tracker with Kalman filtering, class voting and camera motion compensation.
* Monocular ranging (ground contact, size prior, optional Depth Anything) with uncertainty and unbiased lateral estimation.
* Ground-plane motion filtering, time-to-collision, vehicle and robot safety profiles, rate-limited speed governor.
* Lane detection with departure warning, image and ground zones with enter / dwell / exit events.
* Closed-loop procedural simulator (urban and warehouse scenarios) with ground truth.
* Web dashboard (MJPEG, Server-Sent Events, Prometheus), JSONL, video, snapshots and MQTT outputs.
* ROS 2 package (perception node and fail-safe speed governor), CARLA bridge, export, calibration and training scripts.
* Evaluation (CLEAR-MOT, IDF1, range error), latency benchmark, 57 tests, Docker images and CI.
