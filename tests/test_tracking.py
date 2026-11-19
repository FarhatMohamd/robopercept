import numpy as np

from robopercept.config import TrackerConfig
from robopercept.structures import Detection
from robopercept.tracking import BoxKalmanFilter, ByteTracker, PointKalmanFilter


def test_box_kalman_learns_constant_velocity():
    kf = BoxKalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 0.5, 80.0]))
    for k in range(1, 30):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([100.0 + 4 * k, 100.0, 0.5, 80.0]))
    assert abs(mean[4] - 4.0) < 0.3  # vx in px/frame
    d = kf.gating_distance(mean, cov, np.array([[mean[0], mean[1], 0.5, 80.0], [mean[0] + 80, mean[1], 0.5, 80.0]]))
    assert d[0] < d[1]


def test_point_kalman_velocity():
    kf = PointKalmanFilter()
    kf.initiate(np.array([0.0, 30.0]), np.array([0.2, 1.0]))
    rng = np.random.default_rng(1)
    for k in range(1, 60):
        kf.predict(1 / 30)
        z = 30.0 - 6.0 * k / 30 + rng.normal(0, 0.3)
        kf.update(np.array([0.0, z]), np.array([0.2, 0.6]))
    assert abs(kf.mean[3] + 6.0) < 1.0


def _det(x, y, score=0.9, cls=2, name="car"):
    return Detection([x, y, x + 40, y + 60], score, cls, name)


def test_tracker_keeps_identities_through_crossing():
    tr = ByteTracker(TrackerConfig(min_hits=1))
    ids_a, ids_b = set(), set()
    for f in range(40):
        a = _det(50 + 8 * f, 100)
        b = _det(370 - 8 * f, 104, cls=0, name="person")
        out = tr.update([a, b])
        for t in out:
            (ids_a if t.class_name == "car" else ids_b).add(t.track_id)
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_low_score_detections_keep_track_alive():
    tr = ByteTracker(TrackerConfig(min_hits=1))
    seen = set()
    for f in range(30):
        score = 0.3 if 10 <= f < 18 else 0.9  # partially occluded: low confidence
        for t in tr.update([_det(100 + 5 * f, 100, score)]):
            seen.add(t.track_id)
    assert seen == {1}


def test_lost_track_is_recovered_after_short_gap():
    tr = ByteTracker(TrackerConfig(min_hits=1, track_buffer=30))
    seen = set()
    for f in range(40):
        dets = [] if 15 <= f < 22 else [_det(100 + 3 * f, 100)]
        seen |= {t.track_id for t in tr.update(dets)}
    assert seen == {1}


def test_spurious_low_confidence_does_not_spawn_tracks():
    tr = ByteTracker(TrackerConfig())
    for _ in range(10):
        assert tr.update([_det(300, 300, score=0.3)]) == []


def test_class_vote_smooths_flicker():
    tr = ByteTracker(TrackerConfig(min_hits=1))
    out = []
    for f in range(20):
        name, cls = ("truck", 7) if f % 5 == 0 else ("car", 2)
        out = tr.update([Detection([100 + f, 100, 160 + f, 150], 0.9, cls, name)])
    assert out[0].class_name == "car"


def test_camera_motion_compensation_runs(rng):
    cfg = TrackerConfig(camera_motion_compensation=True, min_hits=1)
    tr = ByteTracker(cfg)
    base = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    base = np.ascontiguousarray(np.repeat(np.repeat(base[::8, ::8], 8, 0), 8, 1))
    for f in range(5):
        img = np.roll(base, 2 * f, axis=1)  # the whole scene pans
        out = tr.update([_det(100 + 2 * f, 80)], img)
    assert len(out) == 1 and out[0].track_id == 1
