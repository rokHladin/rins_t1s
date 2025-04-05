#!/usr/bin/env python3

import rclpy
import os
import math
import json
import random

from ament_index_python.packages import get_package_share_directory

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from ultralytics import YOLO


class detect_faces(Node):

    def __init__(self):
        super().__init__('detect_faces')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('device', ''),
            ]
        )

        marker_topic = "/people_marker"

        self.detection_color = (0, 0, 255)
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.scan = None

        self.rgb_image_sub = self.create_subscription(
            Image,
            "/oakd/rgb/preview/image_raw",
            self.rgb_callback,
            qos_profile_sensor_data)

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            "/oakd/rgb/preview/depth/points",
            self.pointcloud_callback,
            qos_profile_sensor_data)

        self.marker_pub = self.create_publisher(
            Marker,
            marker_topic,
            QoSReliabilityPolicy.BEST_EFFORT)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.model = YOLO("yolov8n.pt")

        package_share_dir = get_package_share_directory('dis_tutorial3')
        self.face_groups = []
        self.saved_faces_file = os.path.join(package_share_dir, 'face_positions.json')

        self.faces = []

        self.marker_timer = self.create_timer(1.0, self.publish_face_group_markers)
        self.marker_id_counter = 0

        self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")

    def add_to_face_groups(self, new_point, threshold=0.5):
        for group in self.face_groups:
            for point in group:
                if np.linalg.norm(point - new_point) < threshold:
                    group.append(new_point)
                    return
        # No match: start new group
        self.face_groups.append([new_point])

    def write_faces_to_file(self):
        averaged_faces = []

        for group in self.face_groups:
            avg = np.mean(group, axis=0)
            averaged_faces.append({
                "x": float(avg[0]),
                "y": float(avg[1]),
                "z": float(avg[2]),
            })

        with open(self.saved_faces_file, 'w') as f:
            json.dump(averaged_faces, f, indent=2)

        self.get_logger().info(f"Wrote {len(averaged_faces)} averaged face positions to {self.saved_faces_file}")

    def publish_face_group_markers(self):
        self.marker_id_counter = 0  # Reset IDs so they overwrite previous markers

        for group in self.face_groups:
            if len(group) == 0:
                continue

            avg = np.mean(group, axis=0)

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "faces"
            marker.id = self.marker_id_counter
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(avg[0])
            marker.pose.position.y = float(avg[1])
            marker.pose.position.z = float(avg[2])
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            self.marker_pub.publish(marker)
            self.marker_id_counter += 1

        
    def rgb_callback(self, data):
        self.faces = []

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.get_logger().info("Running inference on image...")

            # run inference
            res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

            # iterate over results
            for x in res:
                bbox = x.boxes.xyxy
                if bbox.nelement() == 0:
                    continue

                self.get_logger().info("Person has been detected!")

                bbox = bbox[0]
                cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)

                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)

                cv_image = cv2.circle(cv_image, (cx, cy), 5, self.detection_color, -1)

                self.faces.append((cx, cy))

            cv2.imshow("image", cv_image)
            key = cv2.waitKey(1)
            if key == 27:
                print("exiting")
                exit()

        except CvBridgeError as e:
            print(e)

    def ransac_plane_fit(self, points, threshold=0.1, max_iterations=500):
        best_inliers = []
        best_normal = None
        best_point = None

        if len(points) < 3:
            return None, None

        points = np.array(points)

        for _ in range(max_iterations):
            # 1. Randomly select 3 distinct points
            sample = points[random.sample(range(len(points)), 3)]
            p1, p2, p3 = sample

            # 2. Compute the normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) == 0:
                continue  # degenerate plane
            normal = normal / np.linalg.norm(normal)

            distances = np.abs((points - p1).dot(normal))
            inliers = points[distances < threshold]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_normal = normal
                best_point = p1  # or mean of inliers later

        if best_normal is not None and len(best_inliers) > 0:
            # Recompute centroid if needed
            best_point = np.mean(best_inliers, axis=0)
            return best_normal, best_point
        else:
            return None, None

    def pointcloud_callback(self, data):
        offset_distance = 0.5
        height = data.height
        width = data.width

        window_size = 10

        a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
        a = a.reshape((height, width, 3))

        for cx, cy in self.faces:
            # 1. Region around face
            region = a[max(0, cy - window_size):cy + window_size,
                       max(0, cx - window_size):cx + window_size, :]

            points = region.reshape(-1, 3)
            points = points[~np.isnan(points).any(axis=1)]

            if len(points) < 30:
                self.get_logger().warn("Not enough points for plane fitting.")
                continue

            # Check if all points lie (approximately) on a line or point
            spread = np.ptp(points, axis=0)  # peak-to-peak = max - min along each axis

            # If the points don't span enough space in at least 2 dimensions, skip
            if np.count_nonzero(spread > 0.01) < 2:
                self.get_logger().warn("Points do not span enough space to define a plane.")
                continue
            
            normal, face_point = self.ransac_plane_fit(points, threshold=0.015, max_iterations=100)

            if normal is None:
                self.get_logger().warn("RANSAC failed to find a valid plane.")
                continue

            # Flip normal if needed
            camera_origin = np.array([0.0, 0.0, 0.0])  # in base_link frame
            to_face = face_point - camera_origin
            to_face = to_face / np.linalg.norm(to_face)

            if np.dot(normal, to_face) > 0:
                normal = -normal

            offset_point = face_point + normal * offset_distance
                            
            offset_msg = PointStamped()
            offset_msg.header.frame_id = "base_link"
            offset_msg.header.stamp = data.header.stamp
            offset_msg.point.x = float(offset_point[0])
            offset_msg.point.y = float(offset_point[1])
            offset_msg.point.z = float(offset_point[2])

            try:
                # Transform to map frame
                transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
                point_world = tf2_geometry_msgs.do_transform_point(offset_msg, transform)

                offset_point_np = np.array([
                    point_world.point.x,
                    point_world.point.y,
                    point_world.point.z
                ])

                self.add_to_face_groups(offset_point_np)

                self.get_logger().info(f"Face (map frame) at {offset_point_np}")

            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {str(e)}")
        
def main():
    print('Face detection node starting.')

    rclpy.init(args=None)
    node = detect_faces()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Saving face positions and shutting down...")
        node.write_faces_to_file()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

