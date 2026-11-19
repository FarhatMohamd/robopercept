"""End-to-end orchestration: source -> detect -> track -> range -> risk -> outputs."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import cv2
import numpy as np

from .config import Config
from .detectors import Detector, build_detector
from .io.sinks import JsonlSink, MqttSink, SnapshotSink, VideoSink
from .io.sources import Frame, FrameSource, SyntheticSource, open_source
from .perception import DistanceEstimator, GroundMotionEstimator, LaneDetector, PinholeCamera, ZoneMonitor
from .perception.depth import DepthEstimator
from .profiling import StageTimer
from .safety import EventBus, RiskAssessor
from .structures import Detection, Event, FrameResult, RiskLevel, Track
from .tracking import ByteTracker
from .viz import BirdEyeView, compose, draw_hud, draw_lanes, draw_track, draw_zones

log = logging.getLogger(__name__)


class Pipeline:
    """Stateful perception pipeline. Feed frames with :meth:`process`."""

    def __init__(self, cfg: Config, detector: Detector | None = None) -> None:
        self.cfg = cfg
        self.detector = detector or build_detector(cfg.detector)
        self.tracker = ByteTracker(cfg.tracker, frame_rate=cfg.source.fps or 30.0)
        self.camera: PinholeCamera | None = None
        self.ranger: DistanceEstimator | None = None
        self.motion = GroundMotionEstimator()
        self.lanes = LaneDetector(cfg.lanes) if cfg.lanes.enabled else None
        self.depth = DepthEstimator(cfg.depth) if cfg.depth.enabled else None
        self._depth_map: np.ndarray | None = None
        self.zones: ZoneMonitor | None = None
        self.risk = RiskAssessor(cfg.safety)
        self.bus = EventBus()
        self.timer = StageTimer()
        self._departing = False
        s = cfg.safety
        self.bev = BirdEyeView(
            corridor_half_width=s.corridor_half_width_m,
            profile=s.profile,
            stop_distance=s.stop_distance_m,
            slow_distance=s.slow_distance_m,
            z_range=50.0 if s.profile == "vehicle" else 15.0,
            x_range=15.0 if s.profile == "vehicle" else 6.0,
        )

    # ----------------------------------------------------------------- setup
    def _lazy_init(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        if self.camera is None or (self.camera.width, self.camera.image_height) != (w, h):
            self.camera = PinholeCamera.from_config(self.cfg.camera, w, h)
            self.ranger = DistanceEstimator(self.camera, self.cfg.distance)
            self.zones = ZoneMonitor(self.cfg.zones, (w, h)) if self.cfg.zones else None

    # ------------------------------------------------------------------ core
    def process(self, frame: Frame) -> FrameResult:
        img = frame.image
        self._lazy_init(img)
        assert self.camera is not None and self.ranger is not None
        timer = self.timer
        res = FrameResult(frame.index, frame.timestamp, img)

        with timer.stage("detect"):
            res.detections = self.detector.detect(img, frame.meta)

        with timer.stage("track"):
            if self.cfg.tracker.enabled:
                tracks = self.tracker.update(res.detections, img)
            else:
                tracks = [
                    Track(i + 1, d.box, d.score, d.class_id, d.class_name)
                    for i, d in enumerate(res.detections)
                ]

        if self.depth is not None and frame.index % max(1, self.cfg.depth.every_n_frames) == 0:
            with timer.stage("depth"):
                self._depth_map = self.depth.infer(img)
        res.depth_map = self._depth_map

        with timer.stage("range"):
            if self.cfg.distance.enabled:
                self._estimate_ranges(tracks, frame.timestamp)
            self.motion.prune({t.track_id for t in tracks})

        if self.lanes is not None:
            with timer.stage("lanes"):
                res.lanes = self.lanes.detect(img)

        with timer.stage("safety"):
            decision, events = self.risk.assess(tracks, frame.index, frame.timestamp)
            if self.zones is not None:
                events += self.zones.update(tracks, frame.index, frame.timestamp)
            departing = bool(res.lanes is not None and res.lanes.departure)
            if departing and not self._departing:
                events.append(
                    Event("lane_departure", frame.index, frame.timestamp,
                          f"lane departure ({res.lanes.offset_m:+.2f} m)", None, RiskLevel.CAUTION)
                )
            self._departing = departing
        res.tracks = tracks
        res.decision = decision
        res.events = events
        for ev in events:
            self.bus.publish(ev)
        res.timings_ms = timer.tick()
        return res

    def _estimate_ranges(self, tracks: list[Track], t: float) -> None:
        assert self.ranger is not None
        depth_vals: dict[int, float] = {}
        if self.depth is not None and self._depth_map is not None:
            raw = {tr.track_id: self.depth.sample(self._depth_map, tr.box) for tr in tracks}
            geo = {}
            for tr in tracks:
                est = self.ranger.estimate(_as_detection(tr))
                if est is not None:
                    geo[tr.track_id] = est.z
            ids = [k for k in raw if k in geo]
            if ids:
                self.depth.fit_scale(np.array([raw[k] for k in ids]), np.array([geo[k] for k in ids]))
            for k, v in raw.items():
                m = self.depth.to_metric(v)
                if m is not None:
                    depth_vals[k] = m
        for tr in tracks:
            est = self.ranger.estimate(_as_detection(tr), depth_vals.get(tr.track_id))
            if est is None:
                continue
            tr.world = self.motion.update(tr, est.x, est.z, est.sigma, t, est.x_sigma)

    # ----------------------------------------------------------- rendering
    def render(self, res: FrameResult) -> np.ndarray:
        out = res.image.copy()
        out_cfg = self.cfg.output
        draw_lanes(out, res.lanes)
        if self.zones is not None:
            active = {z for t in res.tracks for z in t.zones}
            draw_zones(out, self.zones.polys if all(z.frame == "image" for z in self.cfg.zones) else [],
                       [z.name for z in self.cfg.zones], active)
        for tr in sorted(res.tracks, key=lambda t: -(t.world.z if t.world else 0)):
            draw_track(out, tr, out_cfg.draw_trails)
        draw_hud(out, res, self.timer.fps)
        if out_cfg.draw_bev:
            out = compose(out, self.bev.render(res.tracks))
        return out

    # ----------------------------------------------------------------- loop
    def stream(
        self,
        source: FrameSource | None = None,
        on_result: Callable[[FrameResult, np.ndarray], None] | None = None,
    ) -> Iterator[tuple[FrameResult, np.ndarray]]:
        """Iterate over processed frames, handling every configured output."""
        cfg = self.cfg
        if source is None:
            synthetic = cfg.source.uri.startswith("synthetic")
            source = open_source(cfg.source, None if synthetic else cfg.camera)
        if isinstance(source, SyntheticSource):
            # The simulator dictates the camera geometry it renders with
            self.cfg.camera = source.camera_config
            self.camera = None
        out = cfg.output
        video = VideoSink(out.save_video, source.fps) if out.save_video else None
        jsonl = JsonlSink(out.jsonl_log) if out.jsonl_log else None
        snaps = SnapshotSink(out.event_snapshots_dir) if out.event_snapshots_dir else None
        mqtt = MqttSink(cfg.mqtt.host, cfg.mqtt.port, cfg.mqtt.topic) if cfg.mqtt.enabled else None
        if mqtt is not None:
            self.bus.subscribe(mqtt.publish_event)
        try:
            for frame in source:
                res = self.process(frame)
                source.feedback(res.decision.speed_factor)
                vis = self.render(res)
                if video is not None:
                    video.write(vis)
                if jsonl is not None:
                    extra = {"ego_speed": frame.meta["ego_speed"]} if "ego_speed" in frame.meta else None
                    jsonl.write(res, extra)
                if snaps is not None:
                    for ev in res.events:
                        snaps.handle(ev, vis)
                if mqtt is not None:
                    mqtt.publish_frame(res)
                if on_result is not None:
                    on_result(res, vis)
                if out.show:
                    cv2.imshow("RoboPercept", vis)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                yield res, vis
        finally:
            source.close()
            for sink in (video, jsonl, mqtt):
                if sink is not None:
                    sink.close()
            if out.show:
                cv2.destroyAllWindows()

    def run(self, source: FrameSource | None = None, on_result=None) -> dict:
        n = 0
        for _ in self.stream(source, on_result):
            n += 1
        return {"frames": n, "latency_ms": self.timer.summary()}


def _as_detection(tr: Track) -> Detection:
    return Detection(tr.box, tr.score, tr.class_id, tr.class_name)
