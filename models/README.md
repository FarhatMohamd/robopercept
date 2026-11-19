# Models

Weights are not stored in the repository. Ultralytics downloads `*.pt` files
automatically on first use; exported files land here:

```bash
python scripts/export_model.py --model yolo26n.pt --format onnx            # models/yolo26n.onnx
python scripts/export_model.py --model yolo26s.pt --format engine --half   # TensorRT, on the target GPU
```

| File | Backend | Typical target |
|------|---------|----------------|
| `yolo26n.onnx` | `onnx` | CPU robots (Raspberry Pi 5, NUC) |
| `yolo26s.engine` | `ultralytics` | NVIDIA Jetson Orin, RTX |
| `yolo26n_openvino_model/` | `ultralytics` | Intel CPU / iGPU / NPU |
| `rtdetr-l.pt` | `ultralytics` | transformer detector, GPU |
| `yoloe-26s-seg.pt` | `ultralytics` | open-vocabulary text prompts |
