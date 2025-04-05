#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
import math
import numpy as np
import transforms3d.euler
from robot_commander import RobotCommander
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class InspectionNavigator(Node):
    def __init__(self):
        super().__init__('inspection_navigator')

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.sub_markers = self.create_subscription(MarkerArray, '/inspection_markers', self.markers_callback, qos)
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub_rviz = self.create_publisher(MarkerArray, '/inspection_markers', qos)
        self.pub_visited = self.create_publisher(MarkerArray, '/visited_inspection_markers', 10)

        self.occupancy = None
        self.resolution = None
        self.origin = None

        self.robot_pose = None
        self.cam_poses = []
        self.active_goal = None

        self.timer = self.create_timer(1.0, self.loop)
        self.cmdr = RobotCommander()

    def map_callback(self, msg):
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin.position
        grid = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.occupancy = np.ones_like(grid)
        self.occupancy[grid == 100] = 0
        self.occupancy[grid == -1] = -1

    def markers_callback(self, msg):
        self.get_logger().info("📦 Received markers")
        self.cam_poses = []
        cam_map = {}

        for m in msg.markers:
            if m.type == Marker.ARROW and m.color.b > 0.9:
                yaw = self.quaternion_to_yaw(m.pose.orientation)
                cam_map[m.id] = {
                    'pose': (m.pose.position.x, m.pose.position.y, yaw),
                    'targets': [],
                    'seen': set(),
                    'marker_id': m.id
                }

        for m in msg.markers:
            if m.type == Marker.ARROW and m.color.g > 0.9:
                if len(m.points) >= 2:
                    p1 = m.points[0]
                    p2 = m.points[1]
                    nx = p2.x - p1.x
                    ny = p2.y - p1.y
                    norm = math.hypot(nx, ny)
                    if norm == 0:
                        continue
                    nx /= norm
                    ny /= norm
                    marker_id = m.id
                    if cam_map:
                        closest_cam = min(cam_map.values(), key=lambda c: math.hypot(c['pose'][0] - p1.x, c['pose'][1] - p1.y))
                        closest_cam['targets'].append((p1.x, p1.y, nx, ny, marker_id))

        self.cam_poses = list(cam_map.values())
        self.get_logger().info(f"🟦 Loaded {len(self.cam_poses)} camera poses")

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(q)
        self.robot_pose = (x, y, yaw)

    def quaternion_to_yaw(self, q):
        return transforms3d.euler.quat2euler([q.w, q.x, q.y, q.z])[2]

    def loop(self):

        if self.robot_pose is None or not self.cam_poses or self.occupancy is None:
            return

        if self.active_goal:
            any_seen = False
            for i, (tx, ty, nx, ny, marker_id) in enumerate(self.active_goal['targets']):
                if i in self.active_goal['seen']:
                    continue
                if self.is_visible(self.robot_pose, (tx, ty), (nx, ny)):
                    self.active_goal['seen'].add(i)
                    any_seen = True

            if any_seen:
                self.get_logger().info("👀 Some green targets seen. Updating RViz...")
                self.publish_filtered_markers()

            if len(self.active_goal['seen']) == len(self.active_goal['targets']):
                self.get_logger().info("✅ All targets seen. Canceling move.")
                self.cmdr.cancelTask()
                self.publish_visited_markers(self.active_goal)
                self.active_goal = None
                return

            if not self.cmdr.isTaskComplete():
                return
            else:
                self.get_logger().info("🏁 Arrived at goal.")
                self.publish_visited_markers(self.active_goal)
                self.active_goal = None

        if self.cam_poses:
            next_goal = min(self.cam_poses, key=lambda c: self.distance(self.robot_pose, c['pose']))
            self.cam_poses.remove(next_goal)
            self.active_goal = next_goal
            pose = next_goal['pose']
            self.get_logger().info(f"➡️ Going to next pose at {pose}, distance: {self.distance(self.robot_pose, pose):.2f}")
            self.cmdr.goToPose(pose)

    def bresenham(self, x0, y0, x1, y1):
        """Yield integer points on line from (x0, y0) to (x1, y1) using Bresenham's algorithm"""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy


    def is_visible(self, robot_pose, target, normal, min_angle_deg=45):
        self.get_logger().info(f"⛔")

        if self.occupancy is None or self.resolution is None or self.origin is None:
            return False

        rx, ry, ryaw = robot_pose
        tx, ty = target
        nx, ny = normal

        dx = tx - rx
        dy = ty - ry
        dist = math.hypot(dx, dy)
        if dist == 0:
            return False

        # Check angle between robot heading and target vector
        view_angle = math.atan2(dy, dx)
        angle_to_center = abs((ryaw - view_angle + math.pi) % (2 * math.pi) - math.pi)
        if angle_to_center > math.radians(45):
            self.get_logger().info(f"⛔ Angle to target too steep: {math.degrees(angle_to_center):.1f}°")
            return False

        # Check normal angle (for surface-facing inspection)
        dot = (dx * nx + dy * ny) / (dist * math.hypot(nx, ny))
        angle_to_normal = math.acos(dot)
        self.get_logger().info(f"⛔")

        if angle_to_normal < math.radians(min_angle_deg):
            self.get_logger().info(f"⛔ Not steep enough against surface normal: {math.degrees(angle_to_normal):.1f}°")
            return False

        # Check line of sight (Bresenham)
        rx_pix = int((rx - self.origin.x) / self.resolution)
        ry_pix = int((ry - self.origin.y) / self.resolution)
        tx_pix = int((tx - self.origin.x) / self.resolution)
        ty_pix = int((ty - self.origin.y) / self.resolution)

        for x, y in self.bresenham(rx_pix, ry_pix, tx_pix, ty_pix):
            if 0 <= x < self.occupancy.shape[1] and 0 <= y < self.occupancy.shape[0]:
                if self.occupancy[y, x] != 1:
                    self.get_logger().info(f"⛔ Line of sight blocked at ({x}, {y})")
                    return False

        return True

    
    def has_line_of_sight(self, start, end):
        if self.occupancy is None or self.resolution is None or self.origin is None:
            return False

        def to_grid(p):
            gx = int((p[0] - self.origin.x) / self.resolution)
            gy = int((p[1] - self.origin.y) / self.resolution)
            return gx, gy

        x0, y0 = to_grid(start)
        x1, y1 = to_grid(end)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if self.occupancy[y, x] != 1.0:
                    return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if self.occupancy[y, x] != 1.0:
                    return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        return True

    def distance(self, pose1, pose2):
        return math.hypot(pose1[0] - pose2[0], pose1[1] - pose2[1])

    def publish_filtered_markers(self):
        ma = MarkerArray()
        for cam in self.cam_poses + ([self.active_goal] if self.active_goal else []):
            for i in cam['seen']:
                _, _, _, _, marker_id = cam['targets'][i]
                marker = Marker()
                marker.header.frame_id = "map"
                marker.ns = "inspection"
                marker.id = marker_id
                marker.action = Marker.DELETE
                ma.markers.append(marker)
        self.pub_rviz.publish(ma)

    def publish_visited_markers(self, cam):
        ma = MarkerArray()

        for i in cam['seen']:
            tx, ty, *_ = cam['targets'][i]
            m = Marker()
            m.header.frame_id = "map"
            m.ns = "visited"
            m.id = int(tx * 100) + int(ty * 100)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = tx
            m.pose.position.y = ty
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.scale.z = 0.1
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            ma.markers.append(m)

        # Also mark the camera itself
        x, y, _ = cam['pose']
        m = Marker()
        m.header.frame_id = "map"
        m.ns = "visited"
        m.id = int(x * 1000) + int(y * 1000) + 999999
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.scale.x = 0.2
        m.scale.y = 0.2
        m.scale.z = 0.1
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        ma.markers.append(m)

        self.pub_visited.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = InspectionNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

