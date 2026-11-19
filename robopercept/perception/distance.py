"""Monocular range estimation by fusing independent geometric cues.

Two cues are always available from a single bounding box:

* **ground contact**: the bottom edge of the box touches the road, so its row
  intersects the ground plane at a unique depth (exact on flat ground, very
  sensitive near the horizon);
* **size prior**: the apparent height of an object of known class height is
  inversely proportional to its depth (robust far away, biased by
  intra-class variability).

A third, optional cue is a dense depth map (Depth Anything V2/V3). The cues are
combined by inverse-variance weighting, which is the minimum-variance unbiased
estimate when their errors are independent and Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DistanceConfig
from ..structures import Detection
from .camera import PinholeCamera


# Typical object widths (m), used to recover the centre of boxes cut by the image border
CLASS_WIDTHS = {
    "person": 0.5, "pedestrian": 0.5, "bicycle": 0.6, "cyclist": 0.6, "motorcycle": 0.8,
    "car": 1.8, "truck": 2.5, "bus": 2.55, "forklift": 1.2, "traffic cone": 0.4, "pallet": 1.2,
}  # fmt: skip


@dataclass
class RangeEstimate:
    x: float
    z: float
    sigma: float
    cues: dict[str, float]
    x_sigma: float = 0.3


class DistanceEstimator:
    def __init__(self, camera: PinholeCamera, cfg: DistanceConfig | None = None) -> None:
        self.camera = camera
        self.cfg = cfg or DistanceConfig()
        self.heights = {k.lower(): v for k, v in self.cfg.class_heights.items()}
        self.widths = {k.lower(): v for k, v in CLASS_WIDTHS.items()}
        self.pixel_sigma = 1.5

    def _ground_cue(self, det: Detection) -> tuple[float, float] | None:
        cam = self.camera
        u, v = det.bottom_center
        # A box cut by the bottom image border has no visible ground contact
        if cam.image_height and v >= cam.image_height - 2:
            return None
        pt = cam.pixel_to_ground(u, v)
        if pt is None or pt[1] <= 0.3:
            return None
        z = pt[1]
        # First-order propagation of the pixel error through z = f * H / (v - v0)
        sigma_geom = z * z / (cam.fy * cam.height) * self.pixel_sigma
        return z, max(self.cfg.ground_sigma_rel * z, sigma_geom)

    def _size_cue(self, det: Detection) -> tuple[float, float] | None:
        obj_h = self.heights.get(det.class_name.lower())
        if obj_h is None or det.height < 4:
            return None
        if det.box[1] <= 1:  # cut by the top border: apparent height is too small
            return None
        z = self.camera.fy * obj_h / det.height
        return z, self.cfg.size_sigma_rel * z

    def _lateral(self, det: Detection, z: float) -> tuple[float, float]:
        """Lateral offset of the box centre and its standard deviation.

        A box cut by the left or right border no longer contains the object
        centre, so its lateral measurement is returned with a very large
        uncertainty: the motion filter then keeps extrapolating the last
        reliable lateral position instead of following the shrinking box.
        """
        cam = self.camera
        x1, _, x2, _ = (float(v) for v in det.box)
        u = (x1 + x2) / 2.0
        x = (u - cam.cx) / cam.fx * z
        width_m = self.widths.get(det.class_name.lower(), 1.0)
        sigma = 0.1 + 0.25 * width_m + 0.02 * z
        if cam.width and (x1 <= 2.0 or x2 >= cam.width - 2.0):
            sigma = 25.0
        return float(x), float(sigma)

    def estimate(self, det: Detection, depth_z: float | None = None) -> RangeEstimate | None:
        cues: dict[str, tuple[float, float]] = {}
        g = self._ground_cue(det)
        if g is not None:
            cues["ground"] = g
        s = self._size_cue(det)
        if s is not None:
            cues["size"] = s
        if depth_z is not None and np.isfinite(depth_z) and depth_z > 0:
            cues["depth"] = (float(depth_z), 0.10 * float(depth_z))
        if not cues:
            return None

        weights = {k: 1.0 / (sig * sig) for k, (_, sig) in cues.items()}
        wsum = sum(weights.values())
        z = sum(weights[k] * cues[k][0] for k in cues) / wsum
        sigma = float(np.sqrt(1.0 / wsum))
        if len(cues) > 1:
            # Inflate the uncertainty when the cues disagree beyond their errors
            spread = np.std([v for v, _ in cues.values()])
            sigma = float(np.hypot(sigma, 0.5 * spread))
        z = float(min(z, self.cfg.max_range_m))
        x, x_sigma = self._lateral(det, z)
        return RangeEstimate(
            x=x, z=z, sigma=sigma, cues={k: v for k, (v, _) in cues.items()}, x_sigma=x_sigma
        )
