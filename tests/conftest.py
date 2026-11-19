import numpy as np
import pytest

from robopercept.config import CameraConfig
from robopercept.perception.camera import PinholeCamera


@pytest.fixture
def camera() -> PinholeCamera:
    return PinholeCamera.from_config(CameraConfig(hfov_deg=70.0, height_m=1.4, pitch_deg=1.5), 960, 540)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
