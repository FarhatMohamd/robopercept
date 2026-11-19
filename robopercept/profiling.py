"""Latency bookkeeping for each stage of the pipeline."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np


class StageTimer:
    def __init__(self, window: int = 300) -> None:
        self.samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self.current: dict[str, float] = {}
        self._last_tick: float | None = None
        self.fps = 0.0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            self.current[name] = dt
            self.samples[name].append(dt)

    def tick(self) -> dict[str, float]:
        """Close the current frame; returns its per-stage timings in ms."""
        now = time.perf_counter()
        if self._last_tick is not None:
            inst = 1.0 / max(now - self._last_tick, 1e-6)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._last_tick = now
        out = dict(self.current)
        out["total"] = sum(self.current.values())
        self.samples["total"].append(out["total"])
        self.current = {}
        return out

    def summary(self) -> dict[str, dict[str, float]]:
        res = {}
        for name, vals in self.samples.items():
            arr = np.asarray(vals)
            if arr.size:
                res[name] = {
                    "mean": float(arr.mean()),
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                    "max": float(arr.max()),
                }
        return res
