"""Output sinks: annotated video, JSON-lines telemetry and event snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..structures import Event, FrameResult, RiskLevel


class VideoSink:
    def __init__(self, path: str, fps: float) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None

    def write(self, image: np.ndarray) -> None:
        if self.writer is None:
            h, w = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
        self.writer.write(image)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


class JsonlSink:
    """One JSON object per frame, easy to replay, plot or ingest."""

    def __init__(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(p, "w", encoding="utf-8")  # noqa: SIM115 - kept open for streaming

    def write(self, result: FrameResult, extra: dict[str, Any] | None = None) -> None:
        record = result.to_dict()
        if extra:
            record.update(extra)
        self.fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self.fh.close()


class SnapshotSink:
    """Saves the annotated frame whenever a warning or critical event fires."""

    def __init__(self, directory: str, min_level: RiskLevel = RiskLevel.WARNING) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.min_level = min_level

    def handle(self, event: Event, image: np.ndarray) -> Path | None:
        if event.level < self.min_level:
            return None
        path = self.dir / f"{event.frame_index:07d}_{event.kind}.jpg"
        cv2.imwrite(str(path), image)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(event.to_dict(), fh, indent=2)
        return path


class MqttSink:
    """Publishes decisions and events to an MQTT broker (needs ``paho-mqtt``)."""

    def __init__(self, host: str, port: int, topic: str) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("MQTT output needs `pip install robopercept[mqtt]`.") from exc
        self.topic = topic.rstrip("/")
        self.client = mqtt.Client()
        self.client.connect(host, port, keepalive=30)
        self.client.loop_start()

    def publish_frame(self, result: FrameResult) -> None:
        self.client.publish(f"{self.topic}/decision", json.dumps(result.decision.to_dict()), qos=0)

    def publish_event(self, event: Event) -> None:
        self.client.publish(f"{self.topic}/events", json.dumps(event.to_dict()), qos=1)

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
