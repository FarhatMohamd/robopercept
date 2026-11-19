"""Zero-dependency web dashboard (standard library HTTP server).

Endpoints
---------
``GET  /``               dashboard
``GET  /stream.mjpg``    annotated video as Motion-JPEG
``GET  /api/state``      last frame result as JSON
``GET  /api/telemetry``  Server-Sent Events stream (state + events, ~10 Hz)
``GET  /api/stats``      latency percentiles per stage
``GET  /api/config``     active configuration
``GET  /api/health``     liveness probe
``GET  /metrics``        Prometheus exposition format
``POST /api/control``    ``{"paused": true}`` pauses processing
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

import cv2
import numpy as np

from ..structures import FrameResult

log = logging.getLogger(__name__)


class DashboardState:
    """Thread-safe snapshot shared between the pipeline and the HTTP handlers."""

    def __init__(self, jpeg_quality: int = 80, max_events: int = 100) -> None:
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Condition()
        self._jpeg: bytes | None = None
        self._state: dict[str, Any] = {}
        self._seq = 0
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.stats: dict[str, Any] = {}
        self.config: dict[str, Any] = {}
        self.paused = threading.Event()
        self.started = time.time()
        self.frames = 0

    def update(self, res: FrameResult, vis: np.ndarray, stats: dict[str, Any], fps: float) -> None:
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        state = res.to_dict()
        state["fps"] = round(fps, 1)
        if "ego_speed" in res.meta:
            state["ego_speed"] = round(float(res.meta["ego_speed"]), 2)
        with self._lock:
            if ok:
                self._jpeg = buf.tobytes()
            self._state = state
            self.stats = stats
            self.frames += 1
            for ev in state["events"]:
                self.events.appendleft(ev)
            self._seq += 1
            self._lock.notify_all()

    def wait(self, last_seq: int, timeout: float = 1.0) -> tuple[int, bytes | None, dict[str, Any]]:
        with self._lock:
            if self._seq == last_seq:
                self._lock.wait(timeout)
            return self._seq, self._jpeg, self._state

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)


def _make_handler(state: DashboardState) -> type[BaseHTTPRequestHandler]:
    static = resources.files("robopercept.server") / "static"

    class Handler(BaseHTTPRequestHandler):
        server_version = "RoboPercept/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # route to logging
            log.debug("%s - %s", self.address_string(), fmt % args)

        # ------------------------------------------------------------ helpers
        def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        # ------------------------------------------------------------- routes
        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    self._static("index.html", "text/html; charset=utf-8")
                elif path in ("/app.js", "/style.css"):
                    ctype = "application/javascript" if path.endswith(".js") else "text/css"
                    self._static(path[1:], ctype)
                elif path == "/stream.mjpg":
                    self._mjpeg()
                elif path == "/api/telemetry":
                    self._sse()
                elif path == "/api/state":
                    self._json(state.snapshot())
                elif path == "/api/stats":
                    self._json(state.stats)
                elif path == "/api/config":
                    self._json(state.config)
                elif path == "/api/events":
                    self._json(list(state.events))
                elif path == "/api/health":
                    self._json({
                        "status": "ok",
                        "uptime_s": round(time.time() - state.started, 1),
                        "frames": state.frames,
                        "paused": state.paused.is_set(),
                    })
                elif path == "/metrics":
                    self._send(200, _prometheus(state).encode(), "text/plain; version=0.0.4")
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/control":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "invalid JSON"}, HTTPStatus.BAD_REQUEST)
                return
            if "paused" in body:
                (state.paused.set if body["paused"] else state.paused.clear)()
            self._json({"paused": state.paused.is_set()})

        def _static(self, name: str, ctype: str) -> None:
            data = (static / name).read_bytes()
            self._send(200, data, ctype)

        def _mjpeg(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            while True:
                seq, jpeg, _ = state.wait(seq)
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg + b"\r\n")

        def _sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            seq, last_sent = -1, 0.0
            while True:
                seq, _, snap = state.wait(seq)
                now = time.time()
                if now - last_sent < 0.1 and not snap.get("events"):
                    continue
                last_sent = now
                payload = {"state": snap, "stats": state.stats.get("total", {})}
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

    return Handler


def _prometheus(state: DashboardState) -> str:
    snap = state.snapshot()
    lines = [
        "# HELP robopercept_frames_total Processed frames.",
        "# TYPE robopercept_frames_total counter",
        f"robopercept_frames_total {state.frames}",
        "# HELP robopercept_fps Processing rate.",
        "# TYPE robopercept_fps gauge",
        f"robopercept_fps {snap.get('fps', 0)}",
        "# HELP robopercept_tracks Active tracks.",
        "# TYPE robopercept_tracks gauge",
        f"robopercept_tracks {len(snap.get('tracks', []))}",
    ]
    dec = snap.get("decision", {})
    levels = {"safe": 0, "caution": 1, "warning": 2, "critical": 3}
    lines += [
        "# HELP robopercept_risk_level Current risk level (0 safe .. 3 critical).",
        "# TYPE robopercept_risk_level gauge",
        f"robopercept_risk_level {levels.get(dec.get('level', 'safe'), 0)}",
        "# HELP robopercept_speed_factor Speed governor output.",
        "# TYPE robopercept_speed_factor gauge",
        f"robopercept_speed_factor {dec.get('speed_factor', 1.0)}",
        "# HELP robopercept_stage_latency_ms Stage latency percentiles.",
        "# TYPE robopercept_stage_latency_ms gauge",
    ]
    for stage, vals in state.stats.items():
        for q in ("p50", "p95"):
            if q in vals:
                lines.append(f'robopercept_stage_latency_ms{{stage="{stage}",quantile="{q}"}} {vals[q]:.3f}')
    return "\n".join(lines) + "\n"


class DashboardServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, jpeg_quality: int = 80) -> None:
        self.state = DashboardState(jpeg_quality)
        self.httpd = ThreadingHTTPServer((host, port), _make_handler(self.state))
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def start(self) -> DashboardServer:
        self.thread.start()
        log.info("Dashboard on http://localhost:%d", self.port)
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
