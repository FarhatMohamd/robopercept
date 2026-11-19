# ROS 2

The package `ros2_ws/src/robopercept_ros` wraps the library in two nodes.

## Build

```bash
# ROS 2 Jazzy on Ubuntu 24.04
sudo apt install ros-jazzy-vision-msgs ros-jazzy-cv-bridge
pip install --break-system-packages -e .            # from the repository root
pip install --break-system-packages "onnxruntime>=1.17"
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
```

## Perception node

```bash
ros2 run robopercept_ros perception_node --ros-args \
  -p config:=/abs/path/configs/robot_onnx.yaml \
  -p overrides:="['detector.model=/opt/models/yolo26n.onnx']" \
  -r image:=/camera/color/image_raw -r camera_info:=/camera/color/camera_info -r odom:=/odom
```

| Direction | Topic | Type | Notes |
|-----------|-------|------|-------|
| in | `image` | `sensor_msgs/Image` | any encoding supported by `cv_bridge` |
| in | `camera_info` | `sensor_msgs/CameraInfo` | overrides `fx, fy, cx, cy` |
| in | `odom` | `nav_msgs/Odometry` | ego speed for the vehicle profile |
| out | `~/detections` | `vision_msgs/Detection2DArray` | one `Detection2D` per track, `id` = track id, `results[0].pose` = metric position in REP-103 |
| out | `~/markers` | `visualization_msgs/MarkerArray` | cubes coloured by risk plus velocity arrows, in `base_frame` |
| out | `~/speed_factor` | `std_msgs/Float32` | 0 = stop, 1 = nominal |
| out | `~/safety` | `std_msgs/String` | JSON with the decision and the events of the frame |
| out | `~/annotated` | `sensor_msgs/Image` | published only when subscribed |

Parameters: `config` (file path or preset name), `overrides` (list of
`key=value`), `base_frame` (default `base_link`), `publish_annotated`.

The camera mounting height and pitch still come from the configuration file:
they are the two numbers ranging depends on most.

## Speed governor

```bash
ros2 run robopercept_ros speed_governor --ros-args \
  -r speed_factor:=/robopercept/speed_factor -r cmd_vel_in:=/cmd_vel_nav -r cmd_vel_out:=/cmd_vel
```

With Nav2, remap the controller output to `/cmd_vel_nav` (or use the
`velocity_smoother` output) and let the governor publish `/cmd_vel`. Forward
linear velocity is multiplied by the factor, angular velocity by at least 0.3
so the robot can still turn away, reversing is never limited, and everything
stops if perception is silent for `timeout_s`.

## Launch file

```bash
ros2 launch robopercept_ros perception.launch.py \
  image:=/camera/color/image_raw camera_info:=/camera/color/camera_info \
  config:=/abs/path/configs/robot_onnx.yaml cmd_vel_in:=/cmd_vel_nav cmd_vel_out:=/cmd_vel
```
