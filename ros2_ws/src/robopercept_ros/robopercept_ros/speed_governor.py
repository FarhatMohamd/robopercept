"""Velocity governor: scales the navigation command with the perception speed factor.

Insert it between the planner (e.g. Nav2) and the base controller:

    planner --/cmd_vel_nav--> speed_governor --/cmd_vel--> base

It is fail-safe: if no speed factor has been received for ``timeout_s`` (the
perception node crashed or the camera froze) forward motion is stopped.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32

from .conversions import scale_twist


class SpeedGovernor(Node):
    def __init__(self) -> None:
        super().__init__("speed_governor")
        self.declare_parameter("timeout_s", 0.5)
        self.timeout = float(self.get_parameter("timeout_s").value)
        self.factor = 0.0
        self.last = self.get_clock().now()
        self.create_subscription(Float32, "speed_factor", self.on_factor, 10)
        self.create_subscription(Twist, "cmd_vel_in", self.on_cmd, 10)
        self.pub = self.create_publisher(Twist, "cmd_vel_out", 10)

    def on_factor(self, msg: Float32) -> None:
        self.factor = float(msg.data)
        self.last = self.get_clock().now()

    def on_cmd(self, msg: Twist) -> None:
        age = (self.get_clock().now() - self.last).nanoseconds * 1e-9
        factor = self.factor if age <= self.timeout else 0.0
        if age > self.timeout:
            self.get_logger().warn("Perception stale: forward motion inhibited", throttle_duration_sec=2.0)
        (lx, ly, lz), (ax, ay, az) = scale_twist(
            (msg.linear.x, msg.linear.y, msg.linear.z), (msg.angular.x, msg.angular.y, msg.angular.z), factor
        )
        out = Twist()
        out.linear.x, out.linear.y, out.linear.z = lx, ly, lz
        out.angular.x, out.angular.y, out.angular.z = ax, ay, az
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    node = SpeedGovernor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
