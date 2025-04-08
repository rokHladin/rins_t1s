#!/usr/bin/env python3

import rclpy
import math
import numpy as np
import random
from statistics import mode
from collections import defaultdict

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped

from t1s.msg import DetectedRing  # Make sure DetectedRing.msg exists and is built


class RingDetector(Node):
    def __init__(self):
        super().__init__('detect_rings')

        self.ring_pub = self.create_publisher(DetectedRing, "/ring_position", 10)

        self.detected_rings_sent = set()
        self.ring_groups = []

        # Publish mock rings periodically
        self.timer = self.create_timer(2.0, self.mock_ring_detection)
        self.publish_timer = self.create_timer(1.0, self.publish_new_rings)

        self.get_logger().info("🔔 detect_rings running (mock mode)...")

    def mock_ring_detection(self):
        """Mock detection: generate a random ring location near (2, 2)"""
        offset = np.random.uniform(-0.3, 0.3, size=2)
        x, y = 3.5 + offset[0], -1.5 + offset[1]
        color = random.choice(["red", "blue", "green"])

        ring = {
            "point": np.array([x, y, 0.0]),
            "color": color,
        }

        self.add_to_group(ring)

    def add_to_group(self, new_ring, threshold=0.5):
        for group in self.ring_groups:
            if np.linalg.norm(group['point'] - new_ring['point']) < threshold:
                group['points'].append(new_ring['point'])
                group['colors'].append(new_ring['color'])
                return
        self.ring_groups.append({
            'points': [new_ring['point']],
            'colors': [new_ring['color']],
            'point': new_ring['point']
        })

    def publish_new_rings(self):
        for group in self.ring_groups:
            if len(group['points']) < 3:
                continue

            avg_pos = np.mean(group['points'], axis=0)
            key = tuple(np.round(avg_pos, 2))

            if key in self.detected_rings_sent:
                continue

            try:
                color = mode(group['colors'])
            except statistics.StatisticsError:
                color = group['colors'][0]

            msg = DetectedRing()
            msg.position = PointStamped()
            msg.position.header = Header()
            msg.position.header.stamp = self.get_clock().now().to_msg()
            msg.position.header.frame_id = "map"
            msg.position.point.x = float(avg_pos[0])
            msg.position.point.y = float(avg_pos[1])
            msg.position.point.z = float(avg_pos[2])

            msg.color = color

            self.ring_pub.publish(msg)
            self.detected_rings_sent.add(key)

            self.get_logger().info(f"🔔 Published ring at {avg_pos} with color '{color}'")



def main(args=None):
    rclpy.init(args=args)
    node = RingDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

