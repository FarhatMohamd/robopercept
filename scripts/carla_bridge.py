#!/usr/bin/env python3
"""Drive RoboPercept from the CARLA simulator (closed loop).

Start CARLA (0.9.15+), then:
  pip install carla==0.9.15
  python scripts/carla_bridge.py --config configs/vehicle_yolo26.yaml --town Town10HD_Opt

The ego vehicle drives on autopilot; whenever the safety layer asks for a
slow-down, the bridge takes over and brakes proportionally, then gives the
control back to the autopilot.
"""

from __future__ import annotations

import argparse
import math
import queue

import cv2
import numpy as np

from robopercept.config import load_config
from robopercept.io.sources import Frame
from robopercept.pipeline import Pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/vehicle_yolo26.yaml")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 20.0
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    bp_lib = world.get_blueprint_library()
    ego = world.spawn_actor(bp_lib.find("vehicle.tesla.model3"), world.get_map().get_spawn_points()[0])
    ego.set_autopilot(True, tm.get_port())
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(args.width))
    cam_bp.set_attribute("image_size_y", str(args.height))
    cam_bp.set_attribute("fov", str(args.fov))
    mount_h = 1.4
    camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=mount_h)), attach_to=ego)
    images: queue.Queue = queue.Queue()
    camera.listen(images.put)

    cfg = load_config(args.config)
    cfg.source.uri = "carla"
    cfg.camera.hfov_deg = args.fov
    cfg.camera.height_m = mount_h
    cfg.camera.pitch_deg = 0.0
    cfg.camera.fx = cfg.camera.fy = cfg.camera.cx = cfg.camera.cy = None
    pipe = Pipeline(cfg)

    try:
        for i in range(args.frames):
            world.tick()
            img = images.get(timeout=5.0)
            bgr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy()
            v = ego.get_velocity()
            speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            res = pipe.process(Frame(bgr, i, img.timestamp, {"ego_speed": speed}))
            factor = res.decision.speed_factor
            if factor < 0.95:
                ego.set_autopilot(False, tm.get_port())
                ego.apply_control(carla.VehicleControl(throttle=0.0, brake=float(1.0 - factor)))
            else:
                ego.set_autopilot(True, tm.get_port())
            if args.show:
                cv2.imshow("RoboPercept x CARLA", pipe.render(res))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        camera.stop()
        camera.destroy()
        ego.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
