#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion
import transforms3d.euler
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class MockPlanner(Node):
    def __init__(self):
        super().__init__('mock_planner')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(MarkerArray, '/inspection_markers', qos)
        self.timer = self.create_timer(2.0, self.publish_mock_markers)

    def publish_mock_markers(self):
        markers = MarkerArray()

        # 📷 Camera pose (blue arrow)
        cam_x, cam_y, cam_yaw = 3.5, -1.5, math.radians(0)
        cam_quat = transforms3d.euler.euler2quat(0, 0, cam_yaw, axes='sxyz')
        cam_marker = Marker()
        cam_marker.header.frame_id = "map"
        cam_marker.ns = "inspection"
        cam_marker.id = 1
        cam_marker.type = Marker.ARROW
        cam_marker.action = Marker.ADD
        cam_marker.pose.position.x = cam_x
        cam_marker.pose.position.y = cam_y
        cam_marker.pose.orientation = Quaternion(x=cam_quat[1], y=cam_quat[2], z=cam_quat[3], w=cam_quat[0])
        cam_marker.scale.x = 0.5
        cam_marker.scale.y = 0.1
        cam_marker.scale.z = 0.1
        cam_marker.color.r = 0.0
        cam_marker.color.g = 0.0
        cam_marker.color.b = 1.0
        cam_marker.color.a = 1.0
        markers.markers.append(cam_marker)

        # 🎯 Target (green arrow) pointing at the camera
        target_x, target_y = 3.2, -1.5
        target_yaw = math.radians(180)#math.atan2(cam_y - target_y, cam_x - target_x)  # points towards camera
        target_quat = transforms3d.euler.euler2quat(0, 0, target_yaw, axes='sxyz')

        target_marker = Marker()
        target_marker.header.frame_id = "map"
        target_marker.ns = "inspection"
        target_marker.id = 1001
        target_marker.type = Marker.ARROW
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = target_x
        target_marker.pose.position.y = target_y
        target_marker.pose.orientation = Quaternion(x=target_quat[1], y=target_quat[2], z=target_quat[3], w=target_quat[0])
        target_marker.scale.x = 0.2
        target_marker.scale.y = 0.04
        target_marker.scale.z = 0.04
        target_marker.color.r = 0.0
        target_marker.color.g = 1.0
        target_marker.color.b = 0.0
        target_marker.color.a = 1.0
        markers.markers.append(target_marker)

        self.pub.publish(markers)
        self.get_logger().info("📤 Published mock camera + target markers")

def main(args=None):
    rclpy.init(args=args)
    node = MockPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

