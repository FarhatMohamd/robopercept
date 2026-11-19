#!/usr/bin/env python3
"""Export an Ultralytics model for deployment.

Examples
  python scripts/export_model.py --model yolo26n.pt --format onnx --imgsz 640
  python scripts/export_model.py --model yolo26s.pt --format engine --half        # TensorRT (on the target GPU)
  python scripts/export_model.py --model yolo26n.pt --format openvino --int8      # Intel CPU / iGPU / NPU
  python scripts/export_model.py --model yolo26n.pt --format tflite --int8        # Raspberry Pi, Coral
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo26n.pt", help="weights or model name (downloaded if missing)")
    ap.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino", "tflite", "coreml", "ncnn"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 (TensorRT / OpenVINO)")
    ap.add_argument("--int8", action="store_true", help="INT8 post-training quantisation")
    ap.add_argument("--data", default="coco8.yaml", help="calibration data for INT8")
    ap.add_argument("--dynamic", action="store_true", help="dynamic input shape (ONNX)")
    ap.add_argument("--out", default="models", help="destination folder")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install the exporter first: pip install 'robopercept[yolo]'") from exc

    model = YOLO(args.model)
    kwargs = {"format": args.format, "imgsz": args.imgsz, "half": args.half, "int8": args.int8}
    if args.int8:
        kwargs["data"] = args.data
    if args.format == "onnx":
        kwargs.update(dynamic=args.dynamic, simplify=True)
    exported = Path(model.export(**kwargs))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / exported.name
    if exported.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(exported), target)
    print(f"Exported model: {target}")
    print(
        "Use it with:  robopercept run --set detector.backend="
        f"{'onnx' if args.format == 'onnx' else 'ultralytics'} --set detector.model={target}"
    )


if __name__ == "__main__":
    main()
