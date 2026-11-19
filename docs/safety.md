# Safety layer

> RoboPercept is an advisory perception system for research and education.
> It is **not** a certified safety function (ISO 13849, IEC 61508, ISO 26262,
> ISO 3691-4). A machine that can hurt people must keep certified protective
> devices (safety laser scanners, bumpers, emergency stops) independent of it.

## Levels

| Level | Speed factor | Typical cause |
|-------|-------------:|---------------|
| `safe` | 1.0 | nothing relevant |
| `caution` | 0.7 | object in the corridor at medium range, vulnerable road user moving towards it |
| `warning` | 0.4 (vehicle) / ramp (robot) | TTC below `ttc_warning_s`, object inside the slow-down distance |
| `critical` | 0.0 | TTC below `ttc_critical_s`, object inside the stopping distance or the stop distance |

## Vehicle profile (forward-collision warning)

For each track with metric state `(x, z, v_x, v_z)`:

1. `TTC = (z − 0.5) / (−v_z)` if the object closes in and its predicted lateral
   position at that time is inside the corridor
   `|x + v_x·TTC| ≤ corridor_half_width + object_half_width`.
2. Required gap when both brake at `a_max`:
   `d = v_ego·t_r + max(0, v_ego² − v_obj²) / (2 a_max)` with
   `v_obj = v_ego − closing_speed`. This avoids alarms when following a car at
   the same speed while catching a stopped obstacle.
3. `critical` if `TTC < ttc_critical_s` or (in the corridor and closer than
   `max(d, 2 m)`); `warning` if `TTC < ttc_warning_s`; `caution` for objects
   in the corridor within `max(2d, 12 m)`, vulnerable users within 20 m in the
   corridor, or within 10 m and moving towards it.

`v_ego` comes from the frame metadata (simulator, ROS odometry, CARLA) and
falls back to `safety.ego_speed_mps`.

## Robot profile (speed and separation monitoring)

Inspired by the protective-separation logic of ISO/TS 15066 and the speed
limits of ISO 3691-4 for driverless trucks:

* `critical` (protective stop) if an object in the corridor is closer than
  `stop_distance_m`, anything is closer than `0.6 × stop_distance_m`, or
  `TTC < ttc_critical_s`;
* `warning` if an object in the corridor, or any vulnerable agent (people,
  forklifts by default), is within `slow_distance_m`; the speed factor ramps
  linearly from 0.1 at the stop distance to 1 at the slow-down distance;
* `caution` within twice the slow-down distance.

Ground zones (`frame: ground`) are expressed in the robot frame and behave
like the configurable fields of a safety laser scanner.

## Temporal filtering

* **Debounce**: a higher level is reported after `persistence_frames`
  consecutive frames (immediately for `critical`); a lower level after two
  more frames (hysteresis).
* **Rate limiter**: the speed factor drops instantly and recovers at
  0.6 per second, which removes stop-and-go chatter.
* **Event cooldown**: the same collision event for the same track is not
  repeated within 2 s.
* **Fail-safe**: the ROS 2 `speed_governor` stops forward motion when no
  speed factor was received for `timeout_s` (0.5 s).

## Validation

`tests/test_safety.py` covers the TTC geometry (head-on, receding, crossing),
both profiles, debouncing, zones and the event bus.
`tests/test_pipeline.py::test_closed_loop_braking_behind_lead_vehicle` checks
in closed loop that the ego slows below half its speed when the lead vehicle
brakes.
