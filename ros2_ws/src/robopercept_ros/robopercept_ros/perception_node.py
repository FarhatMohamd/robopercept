"""ROS 2 perception node.

Subscriptions
  ``image`` (sensor_msgs/Image), ``camera_info`` (sensor_msgs/CameraInfo, optional),
  ``odom`` (nav_msgs/Odometry, optional: ego speed for the safety layer)

Publications
  ``~/detections``    vision_msgs/Detection2DArray   (tracked objects, ``id`` = track id)
  ``~/markers``       visualization_msgs/MarkerArray (objects in ``base_frame``, for RViz)
  ``~/safety``        std_msgs/String                (JSON decision + events)
  ``~/speed_factor``  std_msgs/Float32               (0 = stop, 1 = nominal)
  ``~/annotated``     sensor_msgs/Image              (optional overlay)
"""

from __future__ import annotations

import json
import math

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from robopercept.config import load_config
from robopercept.io.sources import Frame
from robopercept.pipeline import Pipeline

from .conversions import ego_to_ros, risk_color


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("robopercept")
        self.declare_parameter("config", "")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("overrides", [""])

        cfg_path = self.get_parameter("config").value or None
        overrides = [o for o in self.get_parameter("overrides").value if o]
        self.cfg = load_config(cfg_path, overrides)
        self.cfg.server.enabled = False
        self.cfg.output.show = False
        self.pipeline = Pipeline(self.cfg)
        self.base_frame = self.get_parameter("base_frame").value
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.bridge = CvBridge()
        self.ego_speed: float | None = None
        self.frame_index = 0
        self.t0: float | None = None

        self.create_subscription(Image, "image", self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "camera_info", self.on_camera_info, 1)
        self.create_subscription(Odometry, "odom", self.on_odom, 10)
        self.pub_det = self.create_publisher(Detection2DArray, "~/detections", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "~/markers", 10)
        self.pub_safety = self.create_publisher(String, "~/safety", 10)
        self.pub_speed = self.create_publisher(Float32, "~/speed_factor", 10)
        self.pub_img = self.create_publisher(Image, "~/annotated", 2)
        self.get_logger().info(f"RoboPercept ready, detector backend: {self.cfg.detector.backend}")

    def on_camera_info(self, msg: CameraInfo) -> None:
        k = msg.k
        cam = self.cfg.camera
        if k[0] > 0 and (cam.fx != k[0] or cam.cx != k[2]):
            cam.fx, cam.fy, cam.cx, cam.cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
            self.pipeline.camera = None  # rebuilt on the next frame
            self.get_logger().info("Camera intrinsics updated from CameraInfo")

    def on_odom(self, msg: Odometry) -> None:
        v = msg.twist.twist.linear
        self.ego_speed = math.hypot(v.x, v.y)

    def on_image(self, msg: Image) -> None:
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = stamp
        meta = {} if self.ego_speed is None else {"ego_speed": self.ego_speed}
        res = self.pipeline.process(Frame(img, self.frame_index, stamp - self.t0, meta))
        self.frame_index += 1

        det_array = Detection2DArray(header=msg.header)
        for tr in res.tracks:
            d = Detection2D(header=msg.header)
            d.id = str(tr.track_id)
            x1, y1, x2, y2 = (float(v) for v in tr.box)
            d.bbox.center.position.x = (x1 + x2) / 2.0
            d.bbox.center.position.y = (y1 + y2) / 2.0
            d.bbox.size_x = x2 - x1
            d.bbox.size_y = y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = tr.class_name
            hyp.hypothesis.score = float(tr.score)
            if tr.world is not None:
                fx, ly = ego_to_ros(tr.world.x, tr.world.z)
                hyp.pose.pose.position.x = fx
                hyp.pose.pose.position.y = ly
            d.results.append(hyp)
            det_array.detections.append(d)
        self.pub_det.publish(det_array)
        self.pub_markers.publish(self._markers(res, msg.header.stamp))
        self.pub_speed.publish(Float32(data=float(res.decision.speed_factor)))
        self.pub_safety.publish(
            String(
                data=json.dumps(
                    {
                        "decision": res.decision.to_dict(),
                        "events": [e.to_dict() for e in res.events],
                    }
                )
            )
        )
        if self.publish_annotated and self.pub_img.get_subscription_count() > 0:
            out = self.bridge.cv2_to_imgmsg(self.pipeline.render(res), encoding="bgr8")
            out.header = msg.header
            self.pub_img.publish(out)

    def _markers(self, res, stamp) -> MarkerArray:  # noqa: ANN001
        arr = MarkerArray()
        clear = Marker(action=Marker.DELETEALL)
        clear.header.frame_id = self.base_frame
        arr.markers.append(clear)
        for tr in res.tracks:
            if tr.world is None:
                continue
            m = Marker()
            m.header.frame_id = self.base_frame
            m.header.stamp = stamp
            m.ns = tr.class_name
            m.id = tr.track_id
            m.type = Marker.CUBE
            fx, ly = ego_to_ros(tr.world.x, tr.world.z)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = fx, ly, 0.8
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = 1.0, 0.8, 1.6
            m.color.r, m.color.g, m.color.b, m.color.a = risk_color(tr.risk.label)
            m.lifetime.nanosec = 300_000_000
            arr.markers.append(m)
            v = Marker()
            v.header = m.header
            v.ns = "velocity"
            v.id = tr.track_id
            v.type = Marker.ARROW
            vx, vy = ego_to_ros(tr.world.vx, tr.world.vz)
            v.points = [Point(x=fx, y=ly, z=0.8), Point(x=fx + vx, y=ly + vy, z=0.8)]
            v.scale.x, v.scale.y, v.scale.z = 0.08, 0.2, 0.2
            v.color.r = v.color.g = v.color.b = v.color.a = 1.0
            v.lifetime.nanosec = 300_000_000
            arr.markers.append(v)
        return arr


def main() -> None:
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
