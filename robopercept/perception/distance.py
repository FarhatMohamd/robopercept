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
from .priors import FOOTPRINT


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
        self.footprints = {k.lower(): v for k, v in FOOTPRINT.items()}
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
        """Lateral offset of the object centre and its standard deviation.

        For an object off the optical axis the box also contains its side
        face, so the box centre is biased towards the axis. The *inner*
        vertical edge of the box is the far inner corner of the object, at
        depth ``z + length``; shifting it by half the class width gives an
        unbiased centre. That edge also stays visible when the object leaves
        the image on its outer side, which keeps passing vehicles correctly
        placed until they disappear.
        """
        cam = self.camera
        x1, _, x2, _ = (float(v) for v in det.box)
        fp = self.footprints.get(det.class_name.lower())
        centre = 0.5 * (x1 + x2 - 2 * cam.cx) * z / cam.fx
        sigma = 0.08 + 0.01 * z
        cut_left = cam.width > 0 and x1 <= 3.0
        cut_right = cam.width > 0 and x2 >= cam.width - 3.0
        if fp is None:
            return float(centre), float(sigma + 0.3 + (25.0 if cut_left or cut_right else 0.0))
        half, length = fp[0] / 2.0, fp[1]
        z_far = z + length
        xr = (x2 - cam.cx) * z_far / cam.fx
        xl = (x1 - cam.cx) * z_far / cam.fx
        if xr < -half or (cut_left and not cut_right):  # object on the left of the axis
            return float(xr - half), float(sigma + 0.03 * length)
        if xl > half or (cut_right and not cut_left):  # object on the right
            return float(xl + half), float(sigma + 0.03 * length)
        if cut_left or cut_right:  # straddling the axis and cut: no reliable edge
            return float(centre), 25.0
        return float(centre), float(sigma + 0.1)

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
