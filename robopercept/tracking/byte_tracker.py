"""ByteTrack multi-object tracker (Zhang et al., ECCV 2022), NumPy implementation.

ByteTrack associates *every* detection box, including low-confidence ones: a
first pass matches confident boxes to all tracks, a second pass recovers
occluded objects by matching the remaining low-confidence boxes to the tracks
left over. This keeps identities alive through partial occlusions, which is
the dominant failure mode on the road and in warehouses.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from ..config import TrackerConfig
from ..detectors.nms import iou_matrix
from ..structures import Detection, Track
from .cmc import CameraMotionCompensator, apply_warp_to_state
from .kalman import BoxKalmanFilter
from .matching import class_penalty, fuse_score, linear_assignment


class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(s: np.ndarray) -> np.ndarray:
    w = s[2] * s[3]
    return np.array(
        [s[0] - w / 2, s[1] - s[3] / 2, s[0] + w / 2, s[1] + s[3] / 2], dtype=np.float32
    )


class _STrack:
    """Internal single-target state."""

    def __init__(self, det: Detection, kf: BoxKalmanFilter) -> None:
        self.kf = kf
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(det.box))
        self.det = det
        self.score = det.score
        self.class_id = det.class_id
        self.class_name = det.class_name
        self.track_id = 0
        self.state = TrackState.NEW
        self.activated = False
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.start_frame = 0
        self.frame_id = 0
        self._class_votes: dict[tuple[int, str], float] = {(det.class_id, det.class_name): det.score}

    @property
    def box(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])

    def predict(self) -> None:
        if self.state != TrackState.TRACKED:
            self.mean[7] = 0.0  # freeze height velocity while lost
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection, frame_id: int) -> None:
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(det.box))
        self.det = det
        self.score = det.score
        self.state = TrackState.TRACKED
        self.hits += 1
        self.time_since_update = 0
        self.frame_id = frame_id
        # Score-weighted class voting smooths car/truck style flickering
        key = (det.class_id, det.class_name)
        self._class_votes[key] = self._class_votes.get(key, 0.0) + det.score
        self.class_id, self.class_name = max(self._class_votes, key=self._class_votes.get)


class ByteTracker:
    def __init__(self, cfg: TrackerConfig | None = None, frame_rate: float = 30.0) -> None:
        self.cfg = cfg or TrackerConfig()
        self.kf = BoxKalmanFilter()
        self.max_time_lost = max(1, int(round(frame_rate / 30.0 * self.cfg.track_buffer)))
        self.tracked: list[_STrack] = []
        self.lost: list[_STrack] = []
        self.frame_id = 0
        self._next_id = 1
        self.cmc = CameraMotionCompensator() if self.cfg.camera_motion_compensation else None
        self._history: dict[int, list[tuple[float, float]]] = {}

    def reset(self) -> None:
        self.tracked.clear()
        self.lost.clear()
        self.frame_id = 0
        self._next_id = 1
        self._history.clear()
        if self.cmc is not None:
            self.cmc.reset()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _cost(tracks: list[_STrack], dets: list[Detection], fuse: bool) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)), dtype=np.float32)
        tb = np.stack([t.box for t in tracks])
        db = np.stack([d.box for d in dets])
        cost = 1.0 - iou_matrix(tb, db)
        if fuse:
            cost = fuse_score(cost, np.array([d.score for d in dets]))
        cost = cost + class_penalty([t.class_id for t in tracks], [d.class_id for d in dets])
        return cost

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    # --------------------------------------------------------------------- core
    def update(self, detections: list[Detection], image: np.ndarray | None = None) -> list[Track]:
        cfg = self.cfg
        self.frame_id += 1
        high = [d for d in detections if d.score >= cfg.high_thresh]
        low = [d for d in detections if cfg.low_thresh <= d.score < cfg.high_thresh]

        confirmed = [t for t in self.tracked if t.activated]
        unconfirmed = [t for t in self.tracked if not t.activated]
        pool = confirmed + self.lost
        for t in pool + unconfirmed:
            t.predict()

        if self.cmc is not None and image is not None:
            mask = np.stack([d.box for d in detections]) if detections else None
            warp = self.cmc.estimate(image, mask)
            for t in pool + unconfirmed:
                apply_warp_to_state(t.mean, t.cov, warp)

        activated: list[_STrack] = []
        refound: list[_STrack] = []

        # 1) confident detections vs. every live track
        matches, u_tracks, u_high = linear_assignment(
            self._cost(pool, high, fuse=True), cfg.match_thresh
        )
        for ti, di in matches:
            t = pool[ti]
            was_lost = t.state == TrackState.LOST
            t.update(high[di], self.frame_id)
            (refound if was_lost else activated).append(t)

        # 2) low-confidence detections vs. remaining *tracked* tracks
        remaining = [pool[i] for i in u_tracks if pool[i].state == TrackState.TRACKED]
        matches, u_rem, _ = linear_assignment(
            self._cost(remaining, low, fuse=False), cfg.second_match_thresh
        )
        for ti, di in matches:
            remaining[ti].update(low[di], self.frame_id)
            activated.append(remaining[ti])
        for i in u_rem:
            remaining[i].state = TrackState.LOST

        # 3) unconfirmed tracks (seen once) vs. leftover confident detections
        left_high = [high[i] for i in u_high]
        matches, u_unc, u_new = linear_assignment(
            self._cost(unconfirmed, left_high, fuse=True), 0.7
        )
        for ti, di in matches:
            t = unconfirmed[ti]
            t.update(left_high[di], self.frame_id)
            if t.hits >= cfg.min_hits:
                t.activated = True
            activated.append(t)
        for i in u_unc:
            unconfirmed[i].state = TrackState.REMOVED

        # 4) start new tracks
        for i in u_new:
            det = left_high[i]
            if det.score < cfg.new_track_thresh:
                continue
            t = _STrack(det, self.kf)
            t.track_id = self._new_id()
            t.state = TrackState.TRACKED
            t.start_frame = t.frame_id = self.frame_id
            # Immediately confirmed on the first frame or when min_hits <= 1
            t.activated = self.frame_id == 1 or cfg.min_hits <= 1
            activated.append(t)

        # 5) bookkeeping
        for t in pool:
            if t.state == TrackState.LOST and self.frame_id - t.frame_id > self.max_time_lost:
                t.state = TrackState.REMOVED
        for t in refound:
            t.activated = True

        live = {id(t): t for t in activated + refound}
        self.tracked = [t for t in live.values() if t.state == TrackState.TRACKED]
        self.lost = [
            t
            for t in pool
            if t.state == TrackState.LOST and id(t) not in live
        ]
        self._dedupe()

        out: list[Track] = []
        for t in self.tracked:
            if not t.activated:
                continue
            box = t.box
            cx, by = float((box[0] + box[2]) / 2), float(box[3])
            hist = self._history.setdefault(t.track_id, [])
            hist.append((cx, by))
            del hist[: -cfg.history]
            out.append(
                Track(
                    track_id=t.track_id,
                    box=box,
                    score=t.score,
                    class_id=t.class_id,
                    class_name=t.class_name,
                    age=t.age,
                    hits=t.hits,
                    time_since_update=t.time_since_update,
                    history=list(hist),
                )
            )
        alive = {t.track_id for t in self.tracked + self.lost}
        for tid in list(self._history):
            if tid not in alive:
                del self._history[tid]
        return out

    def _dedupe(self) -> None:
        """Drop lost tracks that overlap an active track (duplicate identities)."""
        if not self.tracked or not self.lost:
            return
        ious = iou_matrix(np.stack([t.box for t in self.lost]), np.stack([t.box for t in self.tracked]))
        self.lost = [t for t, row in zip(self.lost, ious) if row.max() < 0.85]

    @property
    def lost_ids(self) -> list[int]:
        return [t.track_id for t in self.lost]
