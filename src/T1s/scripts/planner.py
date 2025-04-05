#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import cv2
import math
import transforms3d.euler


class Planner(Node):
    def __init__(self):
        super().__init__('inspection_marker_publisher')

        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)
        self.pub_markers = self.create_publisher(MarkerArray, '/inspection_markers', qos)

        self.cam_offset = 0.5
        self.target_offset = 0.1
        self.spacing = 0.3
        self.max_line_length_m = 2.0
        self.map_received = False

    def map_callback(self, msg):
        if self.map_received:
            return
        self.map_received = True

        res = msg.info.resolution
        origin = msg.info.origin.position
        width = msg.info.width
        height = msg.info.height

        grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
        occ = np.ones_like(grid, dtype=np.float32)
        occ[grid == 100] = 0.0
        occ[grid == -1] = -1.0

        edges = cv2.Canny((occ == 0).astype(np.uint8) * 255, 50, 150)
        lines = cv2.HoughLinesP(edges, 0.1, np.pi / 180, threshold=5, minLineLength=5, maxLineGap=5)

        if lines is None:
            self.get_logger().warn("No walls detected.")
            return

        all_lines = []
        for l in lines:
            all_lines.extend(self.split_line(*l[0], self.max_line_length_m, res))

        poses, _ = self.generate_camera_targets(all_lines, occ, res, origin, height, start_target_id=1000)
        markers = self.generate_markers(poses)
        self.pub_markers.publish(markers)
        self.get_logger().info(f"✅ Published {len(markers.markers)} markers")

    def split_line(self, x1, y1, x2, y2, max_len, res):
        max_len_px = max_len / res
        dist = np.hypot(x2 - x1, y2 - y1)
        num = max(1, int(np.ceil(dist / max_len_px)))
        return [[[int(x1 + (x2 - x1) * i / num), int(y1 + (y2 - y1) * i / num),
                  int(x1 + (x2 - x1) * (i + 1) / num), int(y1 + (y2 - y1) * (i + 1) / num)]]
                for i in range(num)]

    def pixel_to_world(self, x_pix, y_pix, res, origin, height):
        x = origin.x + x_pix * res
        y = origin.y + y_pix * res
        return np.array([x, y])

    def is_valid(self, x, y, grid, buf=1):
        x, y = int(round(x)), int(round(y))
        if x < buf or y < buf or x >= grid.shape[1] - buf or y >= grid.shape[0] - buf:
            return False
        region = grid[y - buf:y + buf + 1, x - buf:x + buf + 1]
        return np.all(region == 1.0)

    def generate_camera_targets(self, lines, grid, res, origin, height, start_target_id=1000):
        spacing_px = self.spacing / res
        cam_offset_px = self.cam_offset / res
        target_offset_px = self.target_offset / res
        poses = []
        target_id = start_target_id

        for line in lines:
            x1, y1, x2, y2 = line[0]
            mid_pix = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            mid_world = self.pixel_to_world(mid_pix[0], mid_pix[1], res, origin, height)
            dir_vec = np.array([x2 - x1, y2 - y1], dtype=np.float32)
            if np.linalg.norm(dir_vec) == 0:
                continue
            dir_vec /= np.linalg.norm(dir_vec)
            norm_vec = np.array([-dir_vec[1], dir_vec[0]])
            num_pts = max(1, int(np.linalg.norm([x2 - x1, y2 - y1]) / spacing_px))

            for direction in [1, -1]:
                targets = []
                for i in range(num_pts + 1):
                    alpha = i / num_pts
                    interp = np.array([x1, y1]) * (1 - alpha) + np.array([x2, y2]) * alpha
                    offset = interp + direction * target_offset_px * norm_vec
                    if self.is_valid(offset[0], offset[1], grid):
                        wp = self.pixel_to_world(offset[0], offset[1], res, origin, height)
                        targets.append((wp[0], wp[1], norm_vec[0] * direction, norm_vec[1] * direction, target_id))
                        target_id += 1

                if not targets:
                    continue

                center_offset = mid_pix + direction * cam_offset_px * norm_vec
                if self.is_valid(center_offset[0], center_offset[1], grid):
                    wp = self.pixel_to_world(center_offset[0], center_offset[1], res, origin, height)
                    yaw = math.atan2(mid_world[1] - wp[1], mid_world[0] - wp[0])
                    poses.append({'pose': (wp[0], wp[1], yaw), 'targets': targets})

        return poses, target_id

    def generate_markers(self, cam_targets):
        markers = MarkerArray()
        cam_marker_id = 0

        for entry in cam_targets:
            x, y, yaw = entry['pose']
            q = transforms3d.euler.euler2quat(0, 0, yaw, axes='sxyz')
            q = (q[1], q[2], q[3], q[0])

            cam = Marker()
            cam.header.frame_id = "map"
            cam.ns = "inspection"
            cam.id = cam_marker_id
            cam_marker_id += 1
            cam.type = Marker.ARROW
            cam.action = Marker.ADD
            cam.pose.position.x = x
            cam.pose.position.y = y
            cam.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            cam.scale.x = 0.5
            cam.scale.y = 0.1
            cam.scale.z = 0.1
            cam.color.r = 0.0
            cam.color.g = 0.0
            cam.color.b = 1.0
            cam.color.a = 1.0
            markers.markers.append(cam)

            for tx, ty, nx, ny, tid in entry['targets']:
                yaw = math.atan2(ny, nx)
                q = transforms3d.euler.euler2quat(0, 0, yaw, axes='sxyz')
                m = Marker()
                m.header.frame_id = "map"
                m.ns = "inspection"
                m.id = tid
                m.type = Marker.ARROW
                m.action = Marker.ADD
                m.pose.position.x = tx
                m.pose.position.y = ty
                m.pose.orientation = Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])
                m.scale.x = 0.2
                m.scale.y = 0.04
                m.scale.z = 0.04
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0
                m.color.a = 1.0
                markers.markers.append(m)

        return markers


def main(args=None):
    rclpy.init(args=args)
    node = Planner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

