#!/usr/bin/env python3
"""Fine-tune a real-time detector on a driving or warehouse dataset.

  python scripts/train.py --data datasets/bdd100k.yaml --model yolo26s.pt --epochs 80 --imgsz 960
  python scripts/train.py --data datasets/kitti.yaml   --model yolo26n.pt --epochs 120

Prepare KITTI with ``python scripts/convert_kitti.py`` first. BDD100K must be
converted to YOLO format with the official ``bdd100k`` toolkit.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto batch from GPU memory")
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--name", default="robopercept")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        cos_lr=True,
        close_mosaic=10,
        # Driving scenes: no vertical flips, mild colour jitter, keep small objects
        flipud=0.0,
        fliplr=0.5,
        hsv_v=0.35,
        scale=0.5,
        patience=30,
    )
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map:.4f}  mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
