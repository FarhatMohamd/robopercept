#!/usr/bin/env python3
"""Intrinsic calibration from a chessboard, printed as a RoboPercept ``camera`` section.

Capture 15-30 views of a printed chessboard covering the whole field of view:
  python scripts/calibrate_camera.py --images calib/*.jpg --board 9x6 --square 0.025
or grab them live (press SPACE to keep a view, Q to finish):
  python scripts/calibrate_camera.py --camera 0 --board 9x6 --square 0.025

Mounting height and pitch are measured on the vehicle/robot and added by hand.
"""

from __future__ import annotations

import argparse
import glob

import cv2
import numpy as np
import yaml


def find_corners(gray: np.ndarray, board: tuple[int, int]) -> np.ndarray | None:
    ok, corners = cv2.findChessboardCorners(gray, board, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--camera", type=int)
    ap.add_argument("--board", default="9x6", help="inner corners, columns x rows")
    ap.add_argument("--square", type=float, default=0.025, help="square size (m)")
    ap.add_argument("--height", type=float, default=1.4, help="mounting height above ground (m)")
    ap.add_argument("--pitch", type=float, default=0.0, help="downward pitch (deg)")
    args = ap.parse_args()

    cols, rows = (int(v) for v in args.board.lower().split("x"))
    board = (cols, rows)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square
    obj_pts, img_pts, size = [], [], None

    paths = [p for pattern in args.images for p in glob.glob(pattern)]
    for p in paths:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        size = gray.shape[::-1]
        c = find_corners(gray, board)
        if c is not None:
            obj_pts.append(objp)
            img_pts.append(c)

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            c = find_corners(gray, board)
            vis = frame.copy()
            if c is not None:
                cv2.drawChessboardCorners(vis, board, c, True)
            cv2.putText(vis, f"views: {len(obj_pts)}  SPACE keep  Q finish", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("calibration", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and c is not None:
                obj_pts.append(objp)
                img_pts.append(c)
            elif key in (ord("q"), 27):
                break
        cap.release()
        cv2.destroyAllWindows()

    if len(obj_pts) < 5 or size is None:
        raise SystemExit(f"Need at least 5 valid views, got {len(obj_pts)}")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    w, h = size
    hfov = 2 * np.degrees(np.arctan(w / (2 * K[0, 0])))
    section = {
        "camera": {
            "fx": round(float(K[0, 0]), 2),
            "fy": round(float(K[1, 1]), 2),
            "cx": round(float(K[0, 2]), 2),
            "cy": round(float(K[1, 2]), 2),
            "hfov_deg": round(float(hfov), 2),
            "height_m": args.height,
            "pitch_deg": args.pitch,
        }
    }
    print(f"# {len(obj_pts)} views, RMS reprojection error {rms:.3f} px, image {w}x{h}")
    print(f"# distortion k1..k3, p1, p2: {np.round(dist.ravel(), 5).tolist()}")
    print(yaml.safe_dump(section, sort_keys=False))


if __name__ == "__main__":
    main()
