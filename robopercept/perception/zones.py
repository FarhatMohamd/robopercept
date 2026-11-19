"""Polygonal safety zones (keep-out areas, crosswalks, loading docks...).

A zone is defined either in normalised image coordinates or in metres on the
ground plane. Tracks are tested with their ground contact point; entering,
leaving and dwelling generate events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import ZoneConfig
from ..structures import Event, RiskLevel, Track

_LEVELS = {"caution": RiskLevel.CAUTION, "warning": RiskLevel.WARNING, "critical": RiskLevel.CRITICAL}


@dataclass
class _ZoneState:
    inside_since: dict[int, float] = field(default_factory=dict)
    dwell_reported: set[int] = field(default_factory=set)


class ZoneMonitor:
    def __init__(self, zones: list[ZoneConfig], image_size: tuple[int, int]) -> None:
        self.zones = zones
        w, h = image_size
        self.polys: list[np.ndarray] = []
        for z in zones:
            pts = np.asarray(z.polygon, dtype=np.float32)
            if z.frame == "image" and pts.max() <= 1.0 + 1e-6:
                pts = pts * np.array([w, h], dtype=np.float32)
            self.polys.append(pts.reshape(-1, 1, 2))
        self.state = [_ZoneState() for _ in zones]

    def contains(self, idx: int, track: Track) -> bool:
        z = self.zones[idx]
        if z.classes and track.class_name not in z.classes:
            return False
        if z.frame == "ground":
            if track.world is None:
                return False
            pt = (float(track.world.x), float(track.world.z))
        else:
            pt = (float((track.box[0] + track.box[2]) / 2), float(track.box[3]))
        return cv2.pointPolygonTest(self.polys[idx], pt, False) >= 0

    def update(self, tracks: list[Track], frame_index: int, t: float) -> list[Event]:
        events: list[Event] = []
        for i, zone in enumerate(self.zones):
            st = self.state[i]
            level = _LEVELS[zone.level]
            present: set[int] = set()
            for tr in tracks:
                if not self.contains(i, tr):
                    continue
                present.add(tr.track_id)
                tr.zones.append(zone.name)
                if tr.track_id not in st.inside_since:
                    st.inside_since[tr.track_id] = t
                    if zone.dwell_s <= 0:
                        events.append(
                            Event("zone_enter", frame_index, t,
                                  f"{tr.class_name} #{tr.track_id} entered '{zone.name}'",
                                  tr.track_id, level, {"zone": zone.name})
                        )
                        st.dwell_reported.add(tr.track_id)
                elif (
                    tr.track_id not in st.dwell_reported
                    and t - st.inside_since[tr.track_id] >= zone.dwell_s
                ):
                    st.dwell_reported.add(tr.track_id)
                    events.append(
                        Event("zone_dwell", frame_index, t,
                              f"{tr.class_name} #{tr.track_id} stayed {zone.dwell_s:.0f}s in '{zone.name}'",
                              tr.track_id, level, {"zone": zone.name})
                    )
            for tid in list(st.inside_since):
                if tid not in present:
                    if tid in st.dwell_reported:
                        events.append(
                            Event("zone_exit", frame_index, t, f"#{tid} left '{zone.name}'",
                                  tid, RiskLevel.SAFE, {"zone": zone.name})
                        )
                    st.inside_since.pop(tid)
                    st.dwell_reported.discard(tid)
        return events
