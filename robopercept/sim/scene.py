"""Procedural 3-D scenes rendered through the same pinhole model as the pipeline.

The simulator produces images *and* exact ground truth (boxes, metric
positions, identities). It makes the full stack runnable on any laptop, gives
repeatable end-to-end tests, and closes the loop: the ego speed follows the
speed factor computed by the safety layer, so a braking lead vehicle or a
crossing pedestrian visibly makes the ego slow down.

Scenarios
---------
``urban``      two-lane road, lead car braking hard, oncoming traffic,
               crossing pedestrian, cyclist, parked truck, cones.
``warehouse``  indoor floor with yellow tape lanes, shelving, walking
               operators and a crossing forklift (robot profile).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import CameraConfig
from ..perception.camera import PinholeCamera

VelocityFn = Callable[[float], tuple[float, float]]


@dataclass
class Actor:
    """A box-shaped object moving on the ground plane (world frame, metres)."""

    name: str
    class_name: str
    size: tuple[float, float, float]  # width, height, length
    color: tuple[int, int, int]
    x0: float
    z0: float  # forward offset from the ego at spawn time
    velocity: VelocityFn
    spawn_t: float = 0.0
    lifetime: float = 1e9
    detectable: bool = True
    gt_id: int = 0
    # Runtime state
    x: float = 0.0
    z: float = 0.0
    alive: bool = False
    spawned: bool = False

    def heading(self, t: float) -> float:
        vx, vz = self.velocity(t)
        if abs(vx) + abs(vz) < 0.3:
            return 0.0
        return math.atan2(vx, vz)


def _const(vx: float, vz: float) -> VelocityFn:
    return lambda t: (vx, vz)


def _ramp(t0: float, t1: float, v0: float, v1: float, vx: float = 0.0) -> VelocityFn:
    """Longitudinal speed changing linearly from ``v0`` to ``v1`` between ``t0`` and ``t1``."""

    def fn(t: float) -> tuple[float, float]:
        if t <= t0:
            return vx, v0
        if t >= t1:
            return vx, v1
        a = (t - t0) / (t1 - t0)
        return vx, v0 + a * (v1 - v0)

    return fn


def _piecewise(points: list[tuple[float, float]], vx: float = 0.0) -> VelocityFn:
    ts = np.array([p[0] for p in points])
    vs = np.array([p[1] for p in points])
    return lambda t: (vx, float(np.interp(t, ts, vs)))


@dataclass
class Scenario:
    name: str
    period: float
    ego_speed: float
    camera: CameraConfig
    actors: Callable[[], list[Actor]]
    environment: str = "road"
    lane_xs: tuple[float, ...] = (-5.25, -1.75, 1.75, 5.25)
    road_half_width: float = 7.0
    ego_lateral: Callable[[float], float] = field(default=lambda t: 0.0)


def _urban_actors() -> list[Actor]:
    return [
        # Lead vehicle: cruises, brakes hard, then pulls away (forward-collision case)
        Actor(
            "lead",
            "car",
            (1.8, 1.5, 4.4),
            (60, 60, 200),
            0.0,
            28.0,
            _piecewise([(0, 10.0), (5.0, 10.0), (7.5, 2.0), (13.0, 2.0), (16.0, 14.0)]),
        ),
        Actor("oncoming1", "car", (1.8, 1.45, 4.3), (200, 170, 60), -3.5, 90.0, _const(0.0, -11.0), spawn_t=1.0),
        Actor("oncoming2", "truck", (2.5, 3.2, 8.0), (80, 120, 160), -3.5, 120.0, _const(0.0, -9.0), spawn_t=8.0),
        Actor("cyclist", "bicycle", (0.6, 1.1, 1.8), (170, 80, 170), 3.7, 40.0, _const(0.0, 6.0), spawn_t=9.0),
        Actor("walker", "person", (0.5, 1.65, 0.4), (200, 100, 60), -6.6, 45.0, _const(0.0, 1.2), spawn_t=2.0),
        Actor("parked", "truck", (2.5, 3.2, 7.5), (90, 90, 90), 6.4, 95.0, _const(0.0, 0.0), spawn_t=17.0),
        # Pedestrian crossing in front of the ego (vulnerable road user case)
        Actor(
            "crossing",
            "person",
            (0.5, 1.75, 0.4),
            (40, 140, 240),
            5.0,
            30.0,
            _const(-1.4, 0.0),
            spawn_t=19.0,
            lifetime=9.0,
        ),
        *[
            Actor(
                f"cone{i}",
                "traffic cone",
                (0.4, 0.7, 0.4),
                (30, 120, 250),
                5.4,
                60.0 + 5 * i,
                _const(0.0, 0.0),
                spawn_t=24.0,
            )
            for i in range(4)
        ],
    ]


def _warehouse_actors() -> list[Actor]:
    shelves = [
        Actor(
            f"shelf{side}{i}",
            "shelf",
            (1.2, 2.4, 3.0),
            (70, 90, 120),
            side * 3.4,
            4.0 + 5.0 * i,
            _const(0.0, 0.0),
            detectable=False,
        )
        for side in (-1, 1)
        for i in range(10)
    ]

    def operator1(t: float) -> tuple[float, float]:
        # Walks towards the robot, then steps aside once the robot has stopped
        if t < 6.0:
            return 0.0, -0.7
        if t < 8.5:
            return 0.9, 0.0
        return 0.0, -0.5

    return [
        *shelves,
        Actor("operator1", "person", (0.5, 1.75, 0.4), (40, 170, 240), 0.3, 11.0, operator1),
        Actor(
            "operator2",
            "person",
            (0.5, 1.7, 0.4),
            (220, 120, 70),
            2.6,
            8.0,
            _const(-0.9, 0.0),
            spawn_t=11.0,
            lifetime=7.0,
        ),
        Actor(
            "forklift",
            "forklift",
            (1.2, 2.1, 2.4),
            (30, 160, 230),
            -2.6,
            11.0,
            _const(1.0, 0.0),
            spawn_t=17.0,
            lifetime=6.0,
        ),
        Actor("pallet", "pallet", (1.2, 0.15, 0.8), (60, 130, 170), -2.1, 20.0, _const(0.0, 0.0)),
    ]


SCENARIOS: dict[str, Scenario] = {
    "urban": Scenario(
        name="urban",
        period=32.0,
        ego_speed=10.0,
        camera=CameraConfig(hfov_deg=70.0, height_m=1.4, pitch_deg=1.5),
        actors=_urban_actors,
        ego_lateral=lambda t: 0.7 * math.sin(2 * math.pi * t / 16.0),
    ),
    "warehouse": Scenario(
        name="warehouse",
        period=26.0,
        ego_speed=1.6,
        camera=CameraConfig(hfov_deg=80.0, height_m=0.6, pitch_deg=4.0),
        actors=_warehouse_actors,
        environment="indoor",
        lane_xs=(-1.2, 1.2),
        road_half_width=6.0,
    ),
}


class SyntheticScene:
    """Stateful renderer stepping a scenario at a fixed frame rate."""

    def __init__(
        self,
        scenario: str = "urban",
        width: int = 960,
        height: int = 540,
        fps: float = 30.0,
        camera_cfg: CameraConfig | None = None,
        seed: int = 3,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
        self.sc = SCENARIOS[scenario]
        self.width, self.height, self.fps = width, height, fps
        self.camera = PinholeCamera.from_config(camera_cfg or self.sc.camera, width, height)
        self.rng = np.random.default_rng(seed)
        self.noise = self.rng.normal(0, 6, (height, width, 1)).astype(np.int16)
        self.frame_index = 0
        self.t = 0.0
        self.loop_t = 0.0
        self.s = 0.0  # ego travelled distance
        self.ego_speed = self.sc.ego_speed
        self.speed_factor = 1.0
        self.closed_loop = True
        self._next_gt = 1
        self._reset_actors()

    # ------------------------------------------------------------------ state
    def _reset_actors(self) -> None:
        self.actors = self.sc.actors()
        self.loop_t = 0.0

    def set_speed_factor(self, factor: float) -> None:
        """Closed-loop feedback from the safety layer (0 = stop, 1 = nominal)."""
        self.speed_factor = float(np.clip(factor, 0.0, 1.0))

    def step(self) -> None:
        dt = 1.0 / self.fps
        target = self.sc.ego_speed * (self.speed_factor if self.closed_loop else 1.0)
        dv = np.clip(target - self.ego_speed, -7.0 * dt, 2.5 * dt)
        self.ego_speed = max(0.0, self.ego_speed + dv)
        self.s += self.ego_speed * dt
        self.t += dt
        self.loop_t += dt
        if self.loop_t >= self.sc.period:
            self._reset_actors()
        for a in self.actors:
            if not a.spawned and self.loop_t >= a.spawn_t:
                a.spawned = a.alive = True
                a.x, a.z = a.x0, self.s + a.z0
                a.gt_id = self._next_gt
                self._next_gt += 1
            if not a.alive:
                continue
            vx, vz = a.velocity(self.loop_t)
            a.x += vx * dt
            a.z += vz * dt
            rel = a.z - self.s
            if self.loop_t - a.spawn_t > a.lifetime or rel < -10 or rel > 160:
                a.alive = False
        self.frame_index += 1

    # -------------------------------------------------------------- rendering
    def _ground_poly(self, xs: list[float], zs: list[float], ego_x: float) -> np.ndarray | None:
        x = np.asarray(xs, dtype=np.float64) - ego_x
        z = np.asarray(zs, dtype=np.float64)
        z = np.maximum(z, 0.5)
        uv, valid = self.camera.project_ground(x, z)
        if not valid.all():
            return None
        return uv.astype(np.int32)

    def _background(self, ego_x: float) -> np.ndarray:
        h, w = self.height, self.width
        img = np.empty((h, w, 3), dtype=np.uint8)
        hv = int(np.clip(self.camera.horizon_v, 0, h - 1))
        if self.sc.environment == "road":
            sky_top, sky_bot = np.array([235, 200, 150]), np.array([250, 235, 215])
            ground = (70, 125, 80)
        else:
            sky_top, sky_bot = np.array([90, 90, 95]), np.array([140, 140, 145])
            ground = (110, 115, 118)
        ramp = np.linspace(0, 1, max(hv, 1))[:, None]
        img[:hv] = (sky_top * (1 - ramp) + sky_bot * ramp)[:, None, :].astype(np.uint8)
        img[hv:] = ground
        sc = self.sc
        far = 200.0
        road = self._ground_poly(
            [-sc.road_half_width, sc.road_half_width, sc.road_half_width, -sc.road_half_width],
            [0.5, 0.5, far, far],
            ego_x,
        )
        road_col = (88, 88, 92) if sc.environment == "road" else (150, 152, 150)
        if road is not None:
            cv2.fillPoly(img, [road], road_col)
        # Lane markings: dashed for inner lines, solid for the outer ones
        dash, gap = (3.0, 6.0) if sc.environment == "road" else (1e9, 0.0)
        color = (235, 235, 235) if sc.environment == "road" else (30, 200, 230)
        for i, lx in enumerate(sc.lane_xs):
            solid = sc.environment != "road" or i in (0, len(sc.lane_xs) - 1)
            hw = 0.075
            if solid:
                segs = [(0.5, far)]
            else:
                start = -(self.s % (dash + gap))
                segs = [(z0, z0 + dash) for z0 in np.arange(start, far, dash + gap) if z0 + dash > 0.5]
            for z0, z1 in segs:
                poly = self._ground_poly(
                    [lx - hw, lx + hw, lx + hw, lx - hw], [max(z0, 0.5), max(z0, 0.5), z1, z1], ego_x
                )
                if poly is not None:
                    cv2.fillPoly(img, [poly], color, lineType=cv2.LINE_AA)
        return img

    def _box_corners(self, a: Actor, ego_x: float) -> np.ndarray:
        w, h, length = a.size
        cam_h = self.camera.height
        yaw = a.heading(self.loop_t)
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([[dx, dz] for dz in (0.0, length) for dx in (-w / 2, w / 2)], dtype=np.float64)
        local[:, 1] -= length / 2 if a.class_name in ("person", "bicycle") else 0.0
        rx = local[:, 0] * c + local[:, 1] * s
        rz = -local[:, 0] * s + local[:, 1] * c
        xs = a.x - ego_x + rx
        zs = (a.z - self.s) + rz
        bottom = np.stack([xs, np.full(4, cam_h), zs], 1)
        top = np.stack([xs, np.full(4, cam_h - h), zs], 1)
        return np.vstack([bottom, top])  # indices 0-3 bottom, 4-7 top

    def _draw_actor(self, img: np.ndarray, a: Actor, uv: np.ndarray, depth: np.ndarray) -> None:
        pts = uv.astype(np.int32)
        col = np.array(a.color, dtype=np.float64)
        faces = [
            ([2, 3, 7, 6], 0.55),
            ([0, 2, 6, 4], 0.75),
            ([1, 3, 7, 5], 0.75),
            ([4, 5, 7, 6], 1.1),
            ([0, 1, 5, 4], 0.95),
        ]
        # Painter's algorithm inside the box as well: farthest faces first
        faces.sort(key=lambda f: -float(depth[f[0]].mean()))
        if a.class_name in ("person", "bicycle"):
            faces = []
        for idx, shade in faces:
            cv2.fillPoly(img, [pts[idx]], tuple(int(v) for v in np.clip(col * shade, 0, 255)), cv2.LINE_AA)
        if a.class_name in ("car", "truck", "bus", "forklift"):
            # Windscreen / cabin hint and tail lights on the visible face
            f = pts[[0, 1, 5, 4]].astype(np.float64)
            top_l, top_r, bot_l, bot_r = f[3], f[2], f[0], f[1]
            win = np.array(
                [
                    top_l + 0.15 * (top_r - top_l) + 0.12 * (bot_l - top_l),
                    top_r - 0.15 * (top_r - top_l) + 0.12 * (bot_r - top_r),
                    top_r - 0.15 * (top_r - top_l) + 0.45 * (bot_r - top_r),
                    top_l + 0.15 * (top_r - top_l) + 0.45 * (bot_l - top_l),
                ]
            ).astype(np.int32)
            cv2.fillPoly(img, [win], (60, 50, 40), cv2.LINE_AA)
            for t in (0.12, 0.88):
                p = (bot_l + t * (bot_r - bot_l) + 0.3 * (top_l - bot_l)).astype(int)
                r = max(1, int(abs(bot_r[0] - bot_l[0]) * 0.05))
                cv2.circle(img, tuple(p), r, (40, 40, 230), -1, cv2.LINE_AA)
        elif a.class_name == "person":
            self._draw_person(img, pts, col)
        elif a.class_name == "bicycle":
            self._draw_cyclist(img, pts, col)
        elif a.class_name == "traffic cone":
            for frac in (0.35, 0.65):
                p0 = (pts[0] + frac * (pts[4] - pts[0])).astype(int)
                p1 = (pts[1] + frac * (pts[5] - pts[1])).astype(int)
                cv2.line(img, tuple(p0), tuple(p1), (240, 240, 240), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_person(img: np.ndarray, pts: np.ndarray, col: np.ndarray) -> None:
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        w, h = max(x2 - x1, 2), max(y2 - y1, 4)
        cx = (x1 + x2) / 2
        dark = tuple(int(v) for v in col * 0.55)
        body = tuple(int(v) for v in col)
        leg_w = max(1, int(w * 0.22))
        for off in (-0.2, 0.2):
            lx = int(cx + off * w)
            cv2.rectangle(img, (lx - leg_w // 2, int(y1 + 0.55 * h)), (lx + leg_w // 2, int(y2)), dark, -1)
        cv2.rectangle(img, (int(cx - 0.4 * w), int(y1 + 0.2 * h)), (int(cx + 0.4 * w), int(y1 + 0.6 * h)), body, -1)
        r = max(2, int(0.1 * h))
        cv2.circle(img, (int(cx), int(y1 + r)), r, (150, 180, 220), -1, cv2.LINE_AA)

    @staticmethod
    def _draw_cyclist(img: np.ndarray, pts: np.ndarray, col: np.ndarray) -> None:
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        w, h = max(x2 - x1, 2), max(y2 - y1, 4)
        cx = (x1 + x2) / 2
        r = max(2, int(0.17 * h))
        cv2.circle(img, (int(cx), int(y2 - r)), r, (30, 30, 30), max(1, r // 4), cv2.LINE_AA)
        cv2.rectangle(
            img,
            (int(cx - 0.3 * w), int(y1 + 0.18 * h)),
            (int(cx + 0.3 * w), int(y2 - 1.6 * r)),
            tuple(int(v) for v in col),
            -1,
        )
        hr = max(2, int(0.09 * h))
        cv2.circle(img, (int(cx), int(y1 + hr)), hr, (150, 180, 220), -1, cv2.LINE_AA)

    def render(self) -> tuple[np.ndarray, dict]:
        ego_x = self.sc.ego_lateral(self.t)
        img = self._background(ego_x)
        visible: list[tuple[float, Actor, np.ndarray, np.ndarray, np.ndarray]] = []
        for a in self.actors:
            if not a.alive:
                continue
            corners = self._box_corners(a, ego_x)
            if corners[:, 2].min() < 0.8:
                continue
            uv, valid = self.camera.project(corners)
            if not valid.all():
                continue
            box = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
            if box[2] < 0 or box[0] > self.width or box[3] < 0 or box[1] > self.height:
                continue
            visible.append((float(corners[:, 2].mean()), a, uv, box, corners[:, 2]))
        visible.sort(key=lambda item: -item[0])  # painter's algorithm: far to near
        for _, a, uv, _, depth in visible:
            self._draw_actor(img, a, uv, depth)

        img = np.clip(img.astype(np.int16) + self.noise, 0, 255).astype(np.uint8)

        gt = []
        boxes = [v[3] for v in visible]
        for i, (_, a, _, box, _) in enumerate(visible):
            if not a.detectable:
                continue
            clipped = np.array(
                [max(box[0], 0), max(box[1], 0), min(box[2], self.width - 1), min(box[3], self.height - 1)]
            )
            area = max((clipped[2] - clipped[0]) * (clipped[3] - clipped[1]), 1.0)
            occluded = 0.0
            for other in boxes[i + 1 :]:  # nearer objects drawn on top
                ix = max(0.0, min(clipped[2], other[2]) - max(clipped[0], other[0]))
                iy = max(0.0, min(clipped[3], other[3]) - max(clipped[1], other[1]))
                occluded += ix * iy
            visibility = float(np.clip(1.0 - occluded / area, 0.0, 1.0))
            if visibility < 0.15 or (clipped[3] - clipped[1]) < 4:
                continue
            gt.append(
                {
                    "gt_id": a.gt_id,
                    "class_name": a.class_name,
                    "box": clipped.tolist(),
                    "visibility": visibility,
                    "x": float(a.x - ego_x),
                    "z": float(a.z - self.s - (a.size[2] / 2 if a.class_name in ("person", "bicycle") else 0.0)),
                }
            )
        meta = {
            "ground_truth": gt,
            "ego_speed": self.ego_speed,
            "ego_lateral": ego_x,
            "scenario": self.sc.name,
        }
        return img, meta
