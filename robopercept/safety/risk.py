"""Collision risk assessment and speed governor.

Two profiles share the same machinery:

* ``vehicle``: forward-collision warning based on time-to-collision (TTC) and on
  the stopping distance ``d = v * t_r + v^2 / (2 * a_max)``;
* ``robot``: speed and separation monitoring in the spirit of ISO/TS 15066 and
  ISO 3691-4 for mobile robots: protective stop inside ``stop_distance``,
  linear slow-down up to ``slow_distance``.

The output is advisory. It is not a certified safety function and must never be
the only protection of a real machine (see ``docs/safety.md``).
"""

from __future__ import annotations

import math

import numpy as np

from ..config import SafetyConfig
from ..perception.priors import half_width
from ..structures import Event, RiskLevel, SafetyDecision, Track


def time_to_collision(
    x: float, z: float, vx: float, vz: float, half_width: float, contact_z: float = 0.5
) -> tuple[float, bool]:
    """Constant-velocity TTC with the ego corridor.

    Returns ``(ttc, in_path)``: the time at which the object reaches the
    ego front (``z = contact_z``) while being inside the corridor
    ``|x| <= half_width``, and whether it currently is in the corridor.
    ``ttc`` is ``inf`` when no collision is predicted.
    """
    in_path = abs(x) <= half_width
    if z <= contact_z:
        return (0.0 if in_path else math.inf), in_path
    if vz >= -0.05:  # receding or static relative to ego
        return math.inf, in_path
    t = (z - contact_z) / -vz
    x_at = x + vx * t
    if abs(x_at) <= half_width:
        return float(t), in_path
    return math.inf, in_path


def stopping_distance(speed: float, reaction_s: float, decel: float) -> float:
    return speed * reaction_s + speed * speed / (2.0 * max(decel, 0.1))


class RiskAssessor:
    def __init__(self, cfg: SafetyConfig | None = None) -> None:
        self.cfg = cfg or SafetyConfig()
        self._counters: dict[int, tuple[RiskLevel, int]] = {}
        self._reported: dict[int, RiskLevel] = {}
        self._last_level = RiskLevel.SAFE
        self._last_event_t: dict[tuple[str, int | None], float] = {}
        self.ego_speed = self.cfg.ego_speed_mps
        # Speed factor output: drops instantly, recovers at a bounded rate (per second)
        self.recovery_rate = 0.6
        self._factor = 1.0
        self._factor_t: float | None = None
        self.event_cooldown_s = 2.0

    def _raw_level(self, tr: Track) -> tuple[RiskLevel, float, bool]:
        cfg = self.cfg
        w = tr.world
        if w is None:
            return RiskLevel.SAFE, math.inf, False
        # Corridor widened by half the object footprint
        half = cfg.corridor_half_width_m + half_width(tr.class_name)
        ttc, in_path = time_to_collision(w.x, w.z, w.vx, w.vz, half)
        dist = w.distance
        vulnerable = tr.class_name in cfg.vulnerable_classes
        level = RiskLevel.SAFE

        if cfg.profile == "robot":
            # Speed & separation monitoring: obstacles in the travel corridor, or any
            # vulnerable agent (people) around the robot, scale the speed down
            near = dist <= cfg.stop_distance_m
            if (in_path and near) or dist <= 0.6 * cfg.stop_distance_m or ttc < cfg.ttc_critical_s:
                level = RiskLevel.CRITICAL
            elif (in_path or vulnerable) and dist <= cfg.slow_distance_m:
                level = RiskLevel.WARNING
            elif (in_path or vulnerable) and dist <= 2.0 * cfg.slow_distance_m:
                level = RiskLevel.CAUTION
        else:
            # Safe gap when both brake at the same deceleration (lead object keeps v_obj >= 0)
            v_ego = self.ego_speed
            v_obj = max(0.0, v_ego - w.closing_speed)
            stop_d = v_ego * cfg.reaction_time_s + max(0.0, v_ego**2 - v_obj**2) / (2.0 * cfg.max_decel_mps2)
            if ttc < cfg.ttc_critical_s or (in_path and 0 < dist <= max(stop_d, 2.0)):
                level = RiskLevel.CRITICAL
            elif ttc < cfg.ttc_warning_s:
                level = RiskLevel.WARNING
            elif in_path and (dist < max(2.0 * stop_d, 12.0) or (vulnerable and dist < 20.0)):
                level = RiskLevel.CAUTION
            elif vulnerable and dist < 10.0 and abs(w.vx) > 0.4 and np.sign(w.vx) == -np.sign(w.x):
                level = RiskLevel.CAUTION  # vulnerable user moving towards the corridor
        return level, ttc, in_path

    def _debounce(self, tid: int, level: RiskLevel) -> RiskLevel:
        """Raise a level only after it persisted; critical TTC bypasses the delay."""
        prev, count = self._counters.get(tid, (RiskLevel.SAFE, 0))
        count = count + 1 if level == prev else 1
        self._counters[tid] = (level, count)
        reported = self._reported.get(tid, RiskLevel.SAFE)
        if level > reported and (count >= self.cfg.persistence_frames or level == RiskLevel.CRITICAL):
            reported = level
        elif level < reported and count >= self.cfg.persistence_frames + 2:
            reported = level  # hysteresis on the way down
        self._reported[tid] = reported
        return reported

    def assess(
        self, tracks: list[Track], frame_index: int, t: float, ego_speed: float | None = None
    ) -> tuple[SafetyDecision, list[Event]]:
        """Score every track and aggregate a single decision for the controller.

        ``ego_speed`` (m/s) should come from odometry when available; it falls
        back to the configured nominal speed.
        """
        cfg = self.cfg
        self.ego_speed = cfg.ego_speed_mps if ego_speed is None else float(ego_speed)
        decision = SafetyDecision()
        events: list[Event] = []
        for tr in tracks:
            raw, ttc, in_path = self._raw_level(tr)
            tr.ttc = ttc
            tr.in_ego_path = in_path
            tr.risk = self._debounce(tr.track_id, raw)
            if tr.world is not None and in_path:
                decision.min_distance = min(decision.min_distance, tr.world.distance)
            decision.min_ttc = min(decision.min_ttc, ttc)
            if tr.risk > decision.level or (
                tr.risk == decision.level and tr.risk > RiskLevel.SAFE and ttc < decision.min_ttc + 1e-9
            ):
                decision.level = tr.risk
                decision.critical_track = tr.track_id if tr.risk > RiskLevel.SAFE else None

        # Speed governor
        if decision.level == RiskLevel.CRITICAL:
            decision.speed_factor = 0.0
        elif decision.level == RiskLevel.WARNING:
            if cfg.profile == "robot" and np.isfinite(decision.min_distance):
                span = max(cfg.slow_distance_m - cfg.stop_distance_m, 1e-3)
                decision.speed_factor = float(np.clip((decision.min_distance - cfg.stop_distance_m) / span, 0.1, 1.0))
            else:
                decision.speed_factor = 0.4
        elif decision.level == RiskLevel.CAUTION:
            decision.speed_factor = 0.7

        # Asymmetric rate limiter: slowing down is immediate, speeding up is gradual,
        # which removes the stop-and-go chatter of a flickering detection
        dt = 0.0 if self._factor_t is None else max(0.0, t - self._factor_t)
        self._factor_t = t
        target = decision.speed_factor
        self._factor = target if target <= self._factor else min(target, self._factor + self.recovery_rate * dt)
        decision.speed_factor = self._factor

        crit = next((tr for tr in tracks if tr.track_id == decision.critical_track), None)
        if crit is not None:
            what = f"{crit.class_name} #{crit.track_id}"
            if np.isfinite(crit.ttc):
                decision.reason = f"{what} TTC {crit.ttc:.1f}s"
            elif crit.world is not None:
                decision.reason = f"{what} at {crit.world.distance:.1f} m"
            else:
                decision.reason = what
        if decision.level > self._last_level and decision.level >= RiskLevel.WARNING:
            kind = "collision_critical" if decision.level == RiskLevel.CRITICAL else "collision_warning"
            key = (kind, decision.critical_track)
            if t - self._last_event_t.get(key, -1e9) >= self.event_cooldown_s:
                self._last_event_t[key] = t
                events.append(
                    Event(
                        kind,
                        frame_index,
                        t,
                        decision.reason,
                        decision.critical_track,
                        decision.level,
                        {"speed_factor": decision.speed_factor},
                    )
                )
        self._last_level = decision.level

        live = {tr.track_id for tr in tracks}
        for tid in list(self._counters):
            if tid not in live:
                self._counters.pop(tid, None)
                self._reported.pop(tid, None)
        return decision, events
