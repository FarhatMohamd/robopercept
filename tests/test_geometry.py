import math

import numpy as np
import pytest

from robopercept.config import CameraConfig, DistanceConfig
from robopercept.perception import DistanceEstimator, PinholeCamera
from robopercept.structures import Detection


def test_focal_from_fov():
    cam = PinholeCamera.from_config(CameraConfig(hfov_deg=90.0), 800, 600)
    assert cam.fx == pytest.approx(400.0)
    assert (cam.cx, cam.cy) == (400.0, 300.0)


@pytest.mark.parametrize("pitch", [0.0, 3.0, 8.0])
def test_ground_backprojection_roundtrip(pitch):
    cam = PinholeCamera.from_config(CameraConfig(hfov_deg=70.0, height_m=1.2, pitch_deg=pitch), 960, 540)
    for x, z in [(0.0, 5.0), (-2.5, 12.0), (3.0, 40.0)]:
        uv, valid = cam.project_ground(x, z)
        assert valid.all()
        back = cam.pixel_to_ground(*uv[0])
        assert back is not None
        assert back[0] == pytest.approx(x, abs=1e-6)
        assert back[1] == pytest.approx(z, abs=1e-6)


def test_homography_matches_projection(camera):
    H = camera.ground_homography()
    p = H @ np.array([1.5, 20.0, 1.0])
    uv, _ = camera.project_ground(1.5, 20.0)
    np.testing.assert_allclose(p[:2] / p[2], uv[0], atol=1e-6)


def test_above_horizon_has_no_ground_point(camera):
    assert camera.pixel_to_ground(480, camera.horizon_v - 5) is None


def _project_box(cam, x, z, w, h, length):
    pts = []
    for dz in (0.0, length):
        for dx in (-w / 2, w / 2):
            for y in (cam.height, cam.height - h):
                pts.append((x + dx, y, z + dz))
    uv, _ = cam.project(np.array(pts))
    return [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]


@pytest.mark.parametrize("x,z", [(0.0, 8.0), (0.2, 25.0), (-3.5, 15.0), (3.6, 40.0)])
def test_range_and_lateral_for_cars(camera, x, z):
    est = DistanceEstimator(camera, DistanceConfig())
    box = _project_box(camera, x, z, 1.8, 1.5, 4.4)
    r = est.estimate(Detection(box, 0.9, 2, "car"))
    assert r is not None
    assert abs(r.z - z) / z < 0.03
    assert abs(r.x - x) < 0.15
    assert set(r.cues) == {"ground", "size"}


def test_cut_box_uses_size_cue_only(camera):
    est = DistanceEstimator(camera)
    # bottom edge on the image border: ground contact not visible
    r = est.estimate(Detection([400, 300, 560, camera.image_height], 0.9, 0, "person"))
    assert r is not None and set(r.cues) == {"size"}


def test_dense_depth_cue_is_fused(camera):
    est = DistanceEstimator(camera)
    box = _project_box(camera, 0.0, 20.0, 1.8, 1.5, 4.4)
    r = est.estimate(Detection(box, 0.9, 2, "car"), depth_z=20.0)
    assert "depth" in r.cues and r.z == pytest.approx(20.0, rel=0.03)


def test_unknown_class_without_ground_contact_returns_none(camera):
    est = DistanceEstimator(camera)
    assert est.estimate(Detection([10, 10, 50, camera.image_height], 0.9, 99, "ufo")) is None


def test_depth_scale_fit():
    from robopercept.config import DepthConfig
    from robopercept.perception.depth import DepthEstimator

    de = DepthEstimator(DepthConfig(enabled=True))
    z = np.array([5.0, 10.0, 20.0, 40.0])
    disparity = (1.0 / z - 0.01) / 0.5  # relative map: 1/z = 0.5 d + 0.01
    de.fit_scale(disparity, z)
    assert de.to_metric(float((1 / 15.0 - 0.01) / 0.5)) == pytest.approx(15.0, rel=1e-3)
    depth_map = np.full((100, 100), 7.0, dtype=np.float32)
    assert de.sample(depth_map, np.array([10, 10, 60, 90])) == 7.0
    assert math.isnan(de.sample(depth_map, np.array([10, 10, 10, 10])))
