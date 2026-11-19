"""Overlay rendering: boxes, trails, lanes, zones, HUD and bird's-eye view."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..perception.priors import footprint
from ..structures import FrameResult, LaneEstimate, RiskLevel, Track

# BGR colours
RISK_COLORS = {
    RiskLevel.SAFE: (110, 200, 80),
    RiskLevel.CAUTION: (40, 210, 240),
    RiskLevel.WARNING: (20, 140, 255),
    RiskLevel.CRITICAL: (50, 50, 235),
}
CLASS_COLORS = {
    "person": (255, 170, 60),
    "pedestrian": (255, 170, 60),
    "bicycle": (230, 120, 220),
    "cyclist": (230, 120, 220),
    "motorcycle": (200, 90, 240),
    "car": (90, 200, 255),
    "truck": (70, 140, 255),
    "bus": (60, 110, 230),
    "traffic cone": (40, 130, 250),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def class_color(name: str) -> tuple[int, int, int]:
    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    h = abs(hash(name)) % 180
    hsv = np.uint8([[[h, 160, 230]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.45) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    if y < th + 40:  # would collide with the HUD bar: move the label inside the box
        y = y + th + 8
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), FONT, scale, (20, 20, 20), 1, cv2.LINE_AA)


def draw_lanes(img: np.ndarray, lanes: LaneEstimate | None) -> None:
    if lanes is None or not lanes.valid:
        return
    y0, y1 = lanes.y_range
    ys = np.linspace(y0 + 10, y1, 24)
    xl = np.polyval(lanes.left, ys)
    xr = np.polyval(lanes.right, ys)
    poly = np.concatenate([np.stack([xl, ys], 1), np.stack([xr, ys], 1)[::-1]]).astype(np.int32)
    overlay = img.copy()
    color = (60, 60, 230) if lanes.departure else (120, 200, 80)
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)
    for xs in (xl, xr):
        pts = np.stack([xs, ys], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)


def draw_zones(img: np.ndarray, polys: list[np.ndarray], names: list[str], active: set[str]) -> None:
    overlay = img.copy()
    for poly, name in zip(polys, names):
        color = (60, 60, 230) if name in active else (200, 160, 60)
        cv2.fillPoly(overlay, [poly.astype(np.int32)], color)
        cv2.polylines(img, [poly.astype(np.int32)], True, color, 2, cv2.LINE_AA)
        x, y = poly.reshape(-1, 2).min(axis=0).astype(int)
        _label(img, name, (x, y + 16), color)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)


def draw_track(img: np.ndarray, tr: Track, trails: bool = True) -> None:
    x1, y1, x2, y2 = tr.box.astype(int)
    ccol = class_color(tr.class_name)
    rcol = RISK_COLORS[tr.risk]
    thick = 2 if tr.risk <= RiskLevel.CAUTION else 3
    cv2.rectangle(img, (x1, y1), (x2, y2), rcol if tr.risk > RiskLevel.SAFE else ccol, thick, cv2.LINE_AA)
    if trails and len(tr.history) > 2:
        pts = np.asarray(tr.history, dtype=np.int32)
        for i in range(1, len(pts)):
            a = i / len(pts)
            cv2.line(img, tuple(pts[i - 1]), tuple(pts[i]), ccol, max(1, int(3 * a)), cv2.LINE_AA)
    parts = [f"{tr.class_name} #{tr.track_id}"]
    if tr.world is not None:
        parts.append(f"{tr.world.distance:.1f}m")
    if math.isfinite(tr.ttc) and tr.ttc < 10:
        parts.append(f"TTC {tr.ttc:.1f}s")
    _label(img, " ".join(parts), (x1, y1 - 2), rcol if tr.risk > RiskLevel.SAFE else ccol)


def draw_hud(img: np.ndarray, res: FrameResult, fps: float) -> None:
    h, w = img.shape[:2]
    d = res.decision
    bar = img[0:34].copy()
    cv2.rectangle(bar, (0, 0), (w, 34), (25, 25, 25), -1)
    cv2.addWeighted(bar, 0.75, img[0:34], 0.25, 0, img[0:34])
    color = RISK_COLORS[d.level]
    cv2.rectangle(img, (8, 6), (150, 28), color, -1)
    cv2.putText(img, d.level.name, (16, 23), FONT, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    total = sum(v for k, v in res.timings_ms.items() if k != "total") or res.timings_ms.get("total", 0.0)
    info = f"{fps:5.1f} FPS  {total:5.1f} ms  tracks {len(res.tracks)}  {d.reason}"
    cv2.putText(img, info, (162, 23), FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    # Speed governor gauge
    gx, gw = w - 170, 150
    cv2.rectangle(img, (gx, 10), (gx + gw, 24), (80, 80, 80), 1)
    cv2.rectangle(img, (gx, 10), (gx + int(gw * d.speed_factor), 24), color, -1)
    cv2.putText(img, f"v x{d.speed_factor:.2f}", (gx - 70, 23), FONT, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    if res.lanes is not None and res.lanes.offset_m is not None:
        txt = f"lane offset {res.lanes.offset_m:+.2f} m"
        col = (60, 60, 230) if res.lanes.departure else (235, 235, 235)
        cv2.putText(img, txt, (10, h - 12), FONT, 0.5, col, 1, cv2.LINE_AA)


class BirdEyeView:
    """Top-down occupancy view in the ego frame."""

    def __init__(
        self,
        size: tuple[int, int] = (300, 420),
        x_range: float = 15.0,
        z_range: float = 50.0,
        corridor_half_width: float = 1.3,
        profile: str = "vehicle",
        stop_distance: float = 1.0,
        slow_distance: float = 3.0,
    ) -> None:
        self.w, self.h = size
        self.x_range = x_range
        self.z_range = z_range
        self.corridor = corridor_half_width
        self.profile = profile
        self.stop_d = stop_distance
        self.slow_d = slow_distance

    def to_px(self, x: float, z: float) -> tuple[int, int]:
        u = self.w / 2 + x / self.x_range * (self.w / 2)
        v = self.h - 20 - z / self.z_range * (self.h - 30)
        return int(round(u)), int(round(v))

    def render(self, tracks: list[Track]) -> np.ndarray:
        img = np.full((self.h, self.w, 3), 28, dtype=np.uint8)
        for z in range(0, int(self.z_range) + 1, 10):
            _, v = self.to_px(0, z)
            cv2.line(img, (0, v), (self.w, v), (55, 55, 55), 1)
            cv2.putText(img, f"{z}m", (4, v - 3), FONT, 0.35, (140, 140, 140), 1, cv2.LINE_AA)
        for x in range(-int(self.x_range), int(self.x_range) + 1, 5):
            u, _ = self.to_px(x, 0)
            cv2.line(img, (u, 0), (u, self.h), (45, 45, 45), 1)
        # Ego corridor
        a = self.to_px(-self.corridor, 0)
        b = self.to_px(self.corridor, self.z_range)
        overlay = img.copy()
        cv2.rectangle(overlay, (a[0], b[1]), (b[0], a[1]), (90, 120, 60), -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        ego = self.to_px(0, 0)
        if self.profile == "robot":
            px_per_m = (self.h - 30) / self.z_range
            cv2.circle(img, ego, int(self.slow_d * px_per_m), RISK_COLORS[RiskLevel.WARNING], 1, cv2.LINE_AA)
            cv2.circle(img, ego, int(self.stop_d * px_per_m), RISK_COLORS[RiskLevel.CRITICAL], 1, cv2.LINE_AA)
        cv2.fillPoly(
            img,
            [np.array([(ego[0], ego[1] - 12), (ego[0] - 8, ego[1] + 6), (ego[0] + 8, ego[1] + 6)])],
            (235, 235, 235),
        )
        for tr in tracks:
            if tr.world is None:
                continue
            w = tr.world
            if w.z > self.z_range or abs(w.x) > self.x_range:
                continue
            width, length = footprint(tr.class_name)
            half = width / 2.0
            p1 = self.to_px(w.x - half, w.z)
            p2 = self.to_px(w.x + half, w.z + length)
            col = RISK_COLORS[tr.risk] if tr.risk > RiskLevel.SAFE else class_color(tr.class_name)
            cv2.rectangle(img, (p1[0], p2[1]), (p2[0], p1[1]), col, -1)
            c = self.to_px(w.x, w.z)
            tip = self.to_px(w.x + w.vx, w.z + w.vz)
            cv2.arrowedLine(img, c, tip, (235, 235, 235), 1, cv2.LINE_AA, tipLength=0.3)
            cv2.putText(img, str(tr.track_id), (p2[0] + 3, p2[1] + 8), FONT, 0.35, col, 1, cv2.LINE_AA)
        cv2.putText(img, "BEV", (self.w - 40, 16), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return img


def compose(frame: np.ndarray, bev: np.ndarray | None) -> np.ndarray:
    """Place the BEV panel at the right of the frame (scaled to the frame height)."""
    if bev is None:
        return frame
    h = frame.shape[0]
    scale = h / bev.shape[0]
    bev_r = cv2.resize(bev, (int(bev.shape[1] * scale), h), interpolation=cv2.INTER_AREA)
    return np.hstack([frame, bev_r])
