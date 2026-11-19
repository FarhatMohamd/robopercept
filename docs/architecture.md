# Architecture

RoboPercept is a single-process, single-thread pipeline (plus a capture thread
for live sources and an HTTP thread for the dashboard). Each stage is a plain
Python object with a small interface, so any of them can be replaced.

```
Frame ──▶ Detector.detect() ──▶ ByteTracker.update() ──▶ DistanceEstimator.estimate()
      ──▶ GroundMotionEstimator.update() ──▶ LaneDetector.detect() ──▶ RiskAssessor.assess()
      ──▶ ZoneMonitor.update() ──▶ FrameResult ──▶ sinks / dashboard / ROS 2
```

| Module | Responsibility |
|--------|----------------|
| `io/sources.py` | Camera, file, RTSP/HTTP, image folder, simulator. Live sources are read in a thread that keeps only the newest frame (latency over completeness). |
| `detectors/` | `Detector.detect(image, meta) -> list[Detection]`. Backends are imported lazily, so the core has no deep-learning dependency. |
| `tracking/byte_tracker.py` | Identities. Two-stage ByteTrack association, Kalman prediction in `(cx, cy, aspect, h)` space, optional affine camera motion compensation estimated from background optical flow. |
| `perception/camera.py` | Pinhole model with mounting height and pitch: projection, back-projection on the ground, ground homography. |
| `perception/distance.py` | Monocular range and lateral position with uncertainty. |
| `perception/motion.py` | One `[x, z, vx, vz]` Kalman filter per track with a variable time step. |
| `safety/risk.py` | TTC, risk levels, debouncing, speed governor, collision events. |
| `perception/zones.py`, `perception/lanes.py` | Scene rules and lane-departure warning. |
| `sim/scene.py` | Procedural scenes with ground truth and closed-loop ego dynamics. |

## Coordinate frames

* **Image**: pixels, `u` to the right, `v` downwards.
* **Ego**: metres on the ground plane, `x` to the right, `z` forward, origin
  below the camera. The camera sits at height `H`; a positive pitch tilts it
  towards the ground.
* **ROS (REP-103)**: `x` forward, `y` left. The ROS node converts with
  `x_ros = z`, `y_ros = -x`.

## Monocular ranging

For a pixel `(u, v)` the viewing ray in the ego frame is

```
d = (dx, dy cosθ + sinθ, −dy sinθ + cosθ),   dx = (u − cx)/fx,  dy = (v − cy)/fy
```

and it meets the ground `Y = H` at `t = H / d_y`, giving `x = t·dx` and
`z = t·d_z`. Applied to the bottom edge of a box this is the **ground cue**.
Its standard deviation follows from first-order propagation of a pixel error
`σ_v`: `σ_z ≈ z² σ_v / (f H)`, which explodes near the horizon.

The **size cue** uses a class height prior `h_c`: `z = f h_c / h_px`, with a
relative error set by the intra-class variability (15 % by default).

Both cues, and the optional dense depth sample, are fused by inverse-variance
weighting `ẑ = Σ wᵢ zᵢ / Σ wᵢ`, `wᵢ = 1/σᵢ²`, the minimum-variance unbiased
combination for independent Gaussian errors. When the cues disagree by more
than their own uncertainty, the reported σ is inflated by half their spread so
that the downstream Kalman filter trusts the measurement less.

A cue is discarded when it cannot be valid: the ground cue when the box
touches the bottom border or the contact point is above the horizon, the size
cue when the box touches the top border or the class has no height prior.

### Lateral position

The centre of a box is a biased estimate of the object centre: off the
optical axis the box also contains the side face of the object. The box edge
closest to the axis is the **far inner corner** of the object, at depth
`z + L` where `L` is the class length. The estimator therefore uses
`x = (u_inner − cx)(z + L)/fx ∓ W/2`, with `W, L` from
`perception/priors.py`. The same edge stays visible when the object leaves the
image on its outer side, which keeps passing cars correctly placed. On the
urban scenario this lowers the lateral error from 0.18 m (box centre) to
0.04 m.

### Dense depth (optional)

`depth.enabled: true` runs Depth Anything V2 (or any `depth-estimation` model
of the Hugging Face hub, Depth Anything 3 included) every `every_n_frames`. A
relative model predicts an affine-invariant inverse depth `d`; the pipeline
fits `1/z = a d + b` against the geometric ranges of the current tracks with
Huber-weighted least squares, smooths `(a, b)` over time, and feeds the
converted value as a third cue with 10 % relative error.

## Motion and TTC

Each track's fused measurement updates a constant-velocity Kalman filter with
white acceleration noise (`σ_a = 3 m/s²`). The measured lateral noise is the
box-edge error plus the range error propagated through the bearing; when no
reliable box edge exists the lateral noise is set to 25 m so the filter keeps
extrapolating instead of following a truncated box.

The time-to-collision is the time at which the object reaches the ego front,
`t = (z − z_contact) / (−v_z)`, retained only if the predicted lateral
position at that time, `x + v_x t`, lies inside the corridor widened by half
the object width.

## Limits

* **Flat ground.** The ground cue assumes the object stands on the plane of
  the vehicle. Slopes, speed bumps and pitching under braking bias it; the
  size cue and dense depth mitigate this. A pitch estimate from IMU or lane
  vanishing point can be injected through `camera.pitch_deg`.
* **Class priors.** Children, trailers and unusual vehicles break the size
  prior; the cue fusion and the disagreement inflation limit the damage.
* **Orientation.** The lateral correction assumes objects aligned with the
  road. A vehicle crossing perpendicularly is placed with an error up to half
  its length.
* **Synthetic results.** The simulator validates the geometry, the tracking
  logic and the safety rules under detector-like noise. It does not measure
  the accuracy of a neural detector on real images; use `robopercept.eval`
  with real annotations for that.
