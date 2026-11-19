#!/usr/bin/env python3
"""Convert the KITTI 2-D object benchmark to the YOLO format.

  python scripts/convert_kitti.py --kitti /data/kitti/training --out /data/kitti_yolo --val-split 0.15

KITTI label line: ``type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry``.
"DontCare" regions are dropped, "Van" is merged into "car" and
"Person_sitting" into "person" so the classes match the rest of the project.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2

CLASS_MAP = {
    "Car": "car", "Van": "car", "Truck": "truck", "Pedestrian": "person",
    "Person_sitting": "person", "Cyclist": "bicycle", "Tram": "bus",
}  # fmt: skip
NAMES = ["car", "truck", "person", "bicycle", "bus"]


def convert_label(path: Path, width: int, height: int) -> list[str]:
    lines = []
    for raw in path.read_text().splitlines():
        parts = raw.split()
        if not parts or parts[0] not in CLASS_MAP:
            continue
        x1, y1, x2, y2 = (float(v) for v in parts[4:8])
        cls = NAMES.index(CLASS_MAP[parts[0]])
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        if w > 0 and h > 0:
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kitti", required=True, help="folder containing image_2/ and label_2/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root, out = Path(args.kitti), Path(args.out)
    images = sorted((root / "image_2").glob("*.png"))
    random.Random(args.seed).shuffle(images)
    n_val = int(len(images) * args.val_split)
    for split, subset in (("val", images[:n_val]), ("train", images[n_val:])):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_path in subset:
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            label = convert_label(root / "label_2" / f"{img_path.stem}.txt", w, h)
            shutil.copy(img_path, out / "images" / split / img_path.name)
            (out / "labels" / split / f"{img_path.stem}.txt").write_text("\n".join(label))
    print(f"Wrote {len(images) - n_val} train / {n_val} val images to {out}")


if __name__ == "__main__":
    main()
