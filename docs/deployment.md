# Deployment

## 1. Calibrate the camera

```bash
python scripts/calibrate_camera.py --camera 0 --board 9x6 --square 0.025 --height 1.35 --pitch 1.0
```

Paste the printed `camera:` section in your configuration. Measure the
mounting height with a tape: a 5 cm error on a 1.4 m height is a 3.6 % range
error at every distance.

## 2. Choose and export a model

| Hardware | Suggested model | Export | Backend |
|----------|-----------------|--------|---------|
| NVIDIA Jetson Orin / RTX | `yolo26s.pt` or `rtdetr-l.pt` | `--format engine --half` (on the device) | `ultralytics` |
| Intel CPU, iGPU or NPU | `yolo26n.pt` | `--format openvino --int8` | `ultralytics` |
| Raspberry Pi 5, ARM SBC | `yolo26n.pt`, `imgsz` 320–416 | `--format onnx` or `--format ncnn` | `onnx` / `ultralytics` |
| Any GPU, no PyTorch | any | `--format onnx` + `onnxruntime-gpu` | `onnx` |

```bash
python scripts/export_model.py --model yolo26n.pt --format onnx --imgsz 416
robopercept bench -c configs/robot_onnx.yaml -s 0 --frames 300
```

YOLO26 and YOLOv10 export end-to-end graphs without NMS; RoboPercept detects
this and skips its own NMS.

## 3. Fine-tune on your domain (optional)

```bash
python scripts/convert_kitti.py --kitti /data/kitti/training --out /data/kitti_yolo
python scripts/train.py --data datasets/kitti.yaml --model yolo26s.pt --epochs 120 --imgsz 960
```

For indoor robots, `configs/open_vocabulary.yaml` with YOLOE-26 often avoids
training altogether: list the objects as text prompts and add their heights
under `distance.class_heights`.

## 4. Run

```bash
robopercept run -c configs/vehicle_yolo26.yaml --serve           # dashboard on :8080
docker compose --profile camera up camera                         # containerised, /dev/video0
docker build -f docker/Dockerfile.jetson -t robopercept:jetson .  # Jetson image
```

## 5. Monitor

* Prometheus: scrape `http://<host>:8080/metrics`.
* MQTT: `mqtt.enabled: true` publishes `<topic>/decision` every frame and
  `<topic>/events` for each event (`docker compose --profile mqtt up mqtt`).
* Logs: `output.jsonl_log` writes one JSON line per frame;
  `output.event_snapshots_dir` saves the annotated frame of each warning.

## Performance tips

* Lane detection is the most expensive CPU stage (about 4 ms at 960×540):
  disable it indoors (`lanes.enabled: false`) or lower the resolution.
* Keep `source.drop_frames: true` for live cameras: processing always uses the
  newest frame.
* Run dense depth every 3–5 frames; ranging stays continuous thanks to the
  geometric cues and the Kalman filter.
