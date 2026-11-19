#!/usr/bin/env python3
"""Render the README media (animated GIF and still frames) from the simulator.

python scripts/make_media.py --out docs/media
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2

from robopercept.config import load_config
from robopercept.pipeline import Pipeline


def render(
    scenario: str, frames: int, stills: dict[int, str], out: Path, gif: str | None, gif_range: tuple[int, int]
) -> None:
    cfg = load_config(scenario)
    cfg.source.max_frames = frames
    cfg.server.enabled = False
    pipe = Pipeline(cfg)
    tmp = out / f"_{scenario}_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    for res, vis in pipe.stream():
        i = res.frame_index
        if i in stills:
            cv2.imwrite(str(out / stills[i]), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if gif and gif_range[0] <= i < gif_range[1] and i % 2 == 0:
            cv2.imwrite(str(tmp / f"{i:05d}.png"), vis)
    if gif:
        palette = out / f"_{scenario}_palette.png"
        pattern = str(tmp / "*.png")
        scale = "fps=15,scale=900:-1:flags=lanczos"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-pattern_type",
                "glob",
                "-i",
                pattern,
                "-vf",
                f"{scale},palettegen=max_colors=96",
                str(palette),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                "15",
                "-pattern_type",
                "glob",
                "-i",
                pattern,
                "-i",
                str(palette),
                "-lavfi",
                f"{scale}[x];[x][1:v]paletteuse=dither=bayer",
                str(out / gif),
            ],
            check=True,
        )
        palette.unlink()
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    render(
        "urban", 760, {255: "urban_lead_braking.jpg", 650: "urban_pedestrian.jpg"}, out, "urban_demo.gif", (200, 420)
    )
    render(
        "warehouse",
        640,
        {110: "warehouse_operator.jpg", 545: "warehouse_forklift.jpg"},
        out,
        "warehouse_demo.gif",
        (60, 240),
    )
    print(f"Media written to {out}")


if __name__ == "__main__":
    main()
