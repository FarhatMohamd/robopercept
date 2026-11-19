import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ros2_ws" / "src" / "robopercept_ros"))

from robopercept_ros.conversions import ego_to_ros, risk_color, scale_twist  # noqa: E402


def test_ego_to_rep103():
    assert ego_to_ros(1.5, 10.0) == (10.0, -1.5)


def test_twist_scaling_limits_forward_only():
    (lx, _, _), (_, _, az) = scale_twist((1.0, 0.0, 0.0), (0.0, 0.0, 0.5), 0.0)
    assert lx == 0.0 and az == 0.15
    (lx, _, _), _ = scale_twist((-0.5, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
    assert lx == -0.5  # reversing away from an obstacle stays possible


def test_risk_colors():
    assert risk_color("critical")[0] > risk_color("safe")[0]
