"""Launch the perception node and the velocity governor.

Example:
  ros2 launch robopercept_ros perception.launch.py \
      image:=/camera/color/image_raw camera_info:=/camera/color/camera_info \
      config:=/path/to/robot_onnx.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = {
        "image": "/camera/image_raw",
        "camera_info": "/camera/camera_info",
        "odom": "/odom",
        "config": "warehouse",
        "cmd_vel_in": "/cmd_vel_nav",
        "cmd_vel_out": "/cmd_vel",
    }
    decl = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
    perception = Node(
        package="robopercept_ros",
        executable="perception_node",
        name="robopercept",
        parameters=[{"config": LaunchConfiguration("config"), "base_frame": "base_link"}],
        remappings=[
            ("image", LaunchConfiguration("image")),
            ("camera_info", LaunchConfiguration("camera_info")),
            ("odom", LaunchConfiguration("odom")),
        ],
        output="screen",
    )
    governor = Node(
        package="robopercept_ros",
        executable="speed_governor",
        name="speed_governor",
        parameters=[{"timeout_s": 0.5}],
        remappings=[
            ("speed_factor", "/robopercept/speed_factor"),
            ("cmd_vel_in", LaunchConfiguration("cmd_vel_in")),
            ("cmd_vel_out", LaunchConfiguration("cmd_vel_out")),
        ],
        output="screen",
    )
    return LaunchDescription([*decl, perception, governor])
