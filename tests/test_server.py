import json
import urllib.request

import numpy as np

from robopercept.server import DashboardServer
from robopercept.structures import Detection, FrameResult, Track


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def test_dashboard_endpoints():
    srv = DashboardServer("127.0.0.1", 0).start()
    try:
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        res = FrameResult(3, 0.1, img, detections=[Detection([1, 2, 30, 40], 0.9, 0, "person")])
        res.tracks = [Track(1, np.array([1, 2, 30, 40]), 0.9, 0, "person")]
        res.timings_ms = {"detect": 1.0, "total": 1.0}
        srv.state.update(res, img, {"total": {"p50": 1.0, "p95": 2.0}}, 30.0)

        status, ctype, body = _get(srv.port, "/")
        assert status == 200 and b"RoboPercept" in body
        _, ctype, body = _get(srv.port, "/api/state")
        state = json.loads(body)
        assert state["frame"] == 3 and state["tracks"][0]["class_name"] == "person"
        _, _, body = _get(srv.port, "/api/health")
        assert json.loads(body)["frames"] == 1
        _, ctype, body = _get(srv.port, "/metrics")
        assert "robopercept_tracks 1" in body.decode()
        _, ctype, _ = _get(srv.port, "/app.js")
        assert "javascript" in ctype

        req = urllib.request.Request(
            f"http://127.0.0.1:{srv.port}/api/control", data=b'{"paused": true}', method="POST"
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            assert json.loads(r.read())["paused"] is True
        assert srv.state.paused.is_set()
    finally:
        srv.stop()
