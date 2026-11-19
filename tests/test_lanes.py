from robopercept.config import LaneConfig
from robopercept.perception import LaneDetector
from robopercept.sim import SyntheticScene


def test_lanes_found_and_offset_sign_matches_ego_drift():
    scene = SyntheticScene("urban")
    det = LaneDetector(LaneConfig())
    offsets = []
    for _ in range(150):
        scene.step()
        img, meta = scene.render()
        est = det.detect(img)
        if est.offset_m is not None:
            offsets.append((meta["ego_lateral"], est.offset_m))
    assert len(offsets) > 100
    # The ego drifts sinusoidally; the estimated offset must follow it
    big = [(g, e) for g, e in offsets if abs(g) > 0.3]
    assert big and all((g > 0) == (e > 0) for g, e in big)
    assert max(abs(g - e) for g, e in big) < 0.3


def test_no_markings_means_invalid():
    import numpy as np

    est = LaneDetector().detect(np.full((360, 640, 3), 90, dtype=np.uint8))
    assert not est.valid and est.offset_m is None
