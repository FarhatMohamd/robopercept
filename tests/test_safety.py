import math

import pytest

from robopercept.config import SafetyConfig, ZoneConfig
from robopercept.perception import ZoneMonitor
from robopercept.safety import EventBus, RiskAssessor, stopping_distance, time_to_collision
from robopercept.structures import Event, RiskLevel, Track, WorldState


def _track(tid=1, x=0.0, z=20.0, vx=0.0, vz=-10.0, name="car"):
    t = Track(tid, box=[0, 0, 10, 10], score=0.9, class_id=0, class_name=name)
    t.world = WorldState(x, z, vx, vz)
    return t


def test_ttc_head_on():
    ttc, in_path = time_to_collision(0.0, 20.5, 0.0, -10.0, 1.5)
    assert in_path and ttc == pytest.approx(2.0)


def test_ttc_receding_or_passing():
    assert math.isinf(time_to_collision(0.0, 20.0, 0.0, 2.0, 1.5)[0])
    # oncoming in the adjacent lane never enters the corridor
    assert math.isinf(time_to_collision(-3.5, 20.0, 0.0, -20.0, 2.2)[0])


def test_ttc_crossing_pedestrian_enters_corridor():
    # 4 m to the right, crossing at 1.5 m/s, 12 m ahead, ego closing at 5 m/s
    ttc, in_path = time_to_collision(4.0, 12.0, -1.5, -5.0, 1.6)
    assert not in_path and ttc == pytest.approx((12.0 - 0.5) / 5.0)


def test_stopping_distance():
    assert stopping_distance(10.0, 1.0, 5.0) == pytest.approx(20.0)


def test_vehicle_profile_levels_and_governor():
    ra = RiskAssessor(SafetyConfig(profile="vehicle", persistence_frames=1))
    dec, events = ra.assess([_track(z=10.5, vz=-10.0)], 0, 0.0, ego_speed=10.0)  # TTC 1 s
    assert dec.level == RiskLevel.CRITICAL and dec.speed_factor == 0.0
    assert events and events[0].kind == "collision_critical"
    ra = RiskAssessor(SafetyConfig(profile="vehicle", persistence_frames=1))
    dec, _ = ra.assess([_track(z=60.0, vz=-25.0)], 0, 0.0, ego_speed=5.0)  # TTC ~2.4 s
    assert dec.level == RiskLevel.WARNING and dec.speed_factor == pytest.approx(0.4)
    ra = RiskAssessor(SafetyConfig(profile="vehicle"))
    dec, _ = ra.assess([_track(z=40.0, vz=0.0)], 0, 0.0, ego_speed=10.0)
    assert dec.level == RiskLevel.SAFE and dec.speed_factor == 1.0


def test_following_at_same_speed_is_not_critical():
    ra = RiskAssessor(SafetyConfig(profile="vehicle", persistence_frames=1))
    dec, _ = ra.assess([_track(z=15.0, vz=0.0)], 0, 0.0, ego_speed=10.0)
    assert dec.level < RiskLevel.CRITICAL


def test_robot_profile_speed_and_separation():
    cfg = SafetyConfig(
        profile="robot", stop_distance_m=1.0, slow_distance_m=3.0, persistence_frames=1, corridor_half_width_m=0.4
    )
    ra = RiskAssessor(cfg)
    dec, _ = ra.assess([_track(z=2.0, vz=0.0, name="person")], 0, 0.0)
    assert dec.level == RiskLevel.WARNING and dec.speed_factor == pytest.approx(0.5)
    dec, _ = ra.assess([_track(z=0.8, vz=0.0, name="person")], 1, 0.03)
    assert dec.level == RiskLevel.CRITICAL and dec.speed_factor == 0.0
    # a pallet beside the corridor does not slow the robot down
    ra = RiskAssessor(cfg)
    dec, _ = ra.assess([_track(x=2.5, z=2.0, vz=0.0, name="pallet")], 0, 0.0)
    assert dec.level == RiskLevel.SAFE


def test_debounce_requires_persistence():
    ra = RiskAssessor(SafetyConfig(profile="vehicle", persistence_frames=3))
    levels = []
    for i in range(4):
        dec, _ = ra.assess([_track(z=60.0, vz=-25.0)], i, i / 30, ego_speed=5.0)
        levels.append(dec.level)
    assert levels[:2] == [RiskLevel.SAFE, RiskLevel.SAFE] and levels[2] == RiskLevel.WARNING


def test_zone_enter_dwell_exit():
    zone = ZoneConfig(name="dock", polygon=[(0, 0), (0.5, 0), (0.5, 1), (0, 1)], dwell_s=1.0)
    zm = ZoneMonitor([zone], (100, 100), exit_frames=2)
    inside = Track(7, box=[10, 10, 20, 40], score=0.9, class_id=0, class_name="person")
    outside = Track(7, box=[70, 10, 80, 40], score=0.9, class_id=0, class_name="person")
    kinds = []
    for i in range(40):
        tr = inside if i < 35 else outside
        tr.zones = []
        kinds += [e.kind for e in zm.update([tr], i, i / 30)]
    assert kinds == ["zone_dwell", "zone_exit"]


def test_ground_zone_and_class_filter():
    zone = ZoneConfig(name="field", frame="ground", polygon=[(-1, 0), (1, 0), (1, 3), (-1, 3)], classes=["person"])
    zm = ZoneMonitor([zone], (640, 480))
    person, car = _track(1, 0.2, 2.0, name="person"), _track(2, 0.2, 2.0, name="car")
    events = zm.update([person, car], 0, 0.0)
    assert [e.track_id for e in events] == [1]
    assert person.zones == ["field"] and car.zones == []


def test_event_bus_isolates_failing_subscribers():
    bus = EventBus()
    received = []
    bus.subscribe(lambda e: 1 / 0)
    bus.subscribe(received.append)
    bus.publish(Event("x", 0, 0.0, "hello"))
    assert len(received) == 1 and len(bus.history) == 1
