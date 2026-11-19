"""Frame sources: cameras, video files, network streams, image folders, simulator.

Live sources are read in a background thread that keeps only the newest frame.
Processing therefore always works on the most recent image instead of an
ever-growing backlog, which is what a robot needs (latency over completeness).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import CameraConfig, SourceConfig

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Frame:
    image: np.ndarray
    index: int
    timestamp: float
    meta: dict[str, Any] = field(default_factory=dict)


class FrameSource:
    fps: float = 30.0
    live: bool = False

    def __iter__(self) -> Iterator[Frame]:  # pragma: no cover - interface
        raise NotImplementedError

    def feedback(self, speed_factor: float) -> None:
        """Receive the safety decision (used by the closed-loop simulator)."""

    def close(self) -> None:
        pass


class SyntheticSource(FrameSource):
    def __init__(self, scenario: str, cfg: SourceConfig, camera_cfg: CameraConfig | None = None) -> None:
        from ..sim import SyntheticScene

        w = cfg.width or 960
        h = cfg.height or 540
        self.fps = cfg.fps or 30.0
        self.scene = SyntheticScene(scenario, w, h, self.fps, camera_cfg)
        self.max_frames = cfg.max_frames
        self.loop = True

    @property
    def camera_config(self) -> CameraConfig:
        return self.scene.sc.camera

    def feedback(self, speed_factor: float) -> None:
        self.scene.set_speed_factor(speed_factor)

    def __iter__(self) -> Iterator[Frame]:
        i = 0
        while self.max_frames is None or i < self.max_frames:
            self.scene.step()
            img, meta = self.scene.render()
            yield Frame(img, i, self.scene.t, meta)
            i += 1


class VideoSource(FrameSource):
    def __init__(self, cfg: SourceConfig) -> None:
        uri: Any = cfg.uri
        self.live = uri.isdigit() or uri.startswith(("rtsp://", "rtmp://", "http://", "https://"))
        if uri.isdigit():
            uri = int(uri)
        elif not self.live and not Path(uri).exists():
            raise FileNotFoundError(f"Video source not found: {uri}")
        self.cap = cv2.VideoCapture(uri)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {cfg.uri!r}")
        if cfg.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        if cfg.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        if cfg.fps:
            self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self.fps = cfg.fps or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.cfg = cfg
        self._latest: tuple[int, np.ndarray] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self.live and cfg.drop_frames:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _reader(self) -> None:
        idx = 0
        while not self._stop.is_set():
            ok, img = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = (idx, img)
            idx += 1

    def __iter__(self) -> Iterator[Frame]:
        n = 0
        t0 = time.monotonic()
        last_idx = -1
        while self.cfg.max_frames is None or n < self.cfg.max_frames:
            if self._thread is not None:
                with self._lock:
                    item = self._latest
                if item is None or item[0] == last_idx:
                    time.sleep(0.002)
                    continue
                last_idx, img = item
                yield Frame(img, n, time.monotonic() - t0)
            else:
                ok, img = self.cap.read()
                if not ok:
                    if self.cfg.loop and not self.live:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                ts = n / self.fps if not self.live else time.monotonic() - t0
                yield Frame(img, n, ts)
            n += 1

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()


class ImageFolderSource(FrameSource):
    def __init__(self, cfg: SourceConfig) -> None:
        self.paths = sorted(p for p in Path(cfg.uri).iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {cfg.uri}")
        self.fps = cfg.fps or 10.0
        self.cfg = cfg

    def __iter__(self) -> Iterator[Frame]:
        n = 0
        while True:
            for p in self.paths:
                if self.cfg.max_frames is not None and n >= self.cfg.max_frames:
                    return
                img = cv2.imread(str(p))
                if img is None:
                    continue
                yield Frame(img, n, n / self.fps, {"path": str(p)})
                n += 1
            if not self.cfg.loop:
                return


def open_source(cfg: SourceConfig, camera_cfg: CameraConfig | None = None) -> FrameSource:
    uri = cfg.uri
    if uri == "synthetic" or uri.startswith("synthetic:"):
        scenario = uri.split(":", 1)[1] if ":" in uri else "urban"
        return SyntheticSource(scenario, cfg, camera_cfg)
    if Path(uri).is_dir():
        return ImageFolderSource(cfg)
    return VideoSource(cfg)
