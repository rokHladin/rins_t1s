#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs.tf2_geometry_msgs

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from t1s.msg import DetectedRing
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from rclpy.qos import qos_profile_sensor_data



qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)



def generate_hsv_color_hist(bgr_color, image_size=(100, 100), std_dev=7):
    color_patch = np.full((image_size[1], image_size[0], 3), bgr_color, dtype=np.uint8)
    hsv_patch = cv2.cvtColor(color_patch, cv2.COLOR_BGR2HSV)
    hue_value = int(hsv_patch[0, 0, 0])
    bins = np.arange(180)
    gaussian_hist = np.exp(-0.5 * ((bins - hue_value) / std_dev) ** 2)
    gaussian_hist = gaussian_hist.astype(np.float32)
    gaussian_hist = cv2.normalize(gaussian_hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    gaussian_hist = gaussian_hist.reshape(-1, 1)
    return gaussian_hist


class RingDetector(Node):
    def __init__(self):
        super().__init__('ring_detector')

        # Basic ROS stuff
        timer_frequency = 2
        timer_period = 1/timer_frequency

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1

        # Store the latest depth image
        self.latest_depth = None
        
        self.latest_pointcloud = None
        self.pc_sub = self.create_subscription(
            PointCloud2,
            "/oakd/rgb/preview/depth/points",
            self.pc_callback,
            qos_profile_sensor_data
        )

        
        # Subscribe to the image and/or depth topic
        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
        
        # Publisher for ring detections
        self.ring_pub = self.create_publisher(Marker, "/detected_ring", 10)
        self.ring_info_pub = self.create_publisher(String, "/ring_info", 10)

        # Object we use for transforming between coordinate frames
        # Object we use for transforming between coordinate frames
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ring_groups = []
        self.detected_rings_sent = set()
        self.detected_ring_pub = self.create_publisher(DetectedRing, "/ring_position", 10)
        self.timer = self.create_timer(1.0, self.publish_new_rings)

        # Create windows for visualization
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)

        self.ring_groups = []

        
        # Parameters for ring detection
        self.min_contour_points = 20
        self.max_center_distance = 5.0
        self.max_angle_diff = 7.0
        self.min_ring_width = 2
        self.max_ring_width = 50
        self.min_circle_height = 4
        
        # Parameters for 3D validation
        self.depth_threshold = 1.5  # Expected depth difference for 3D rings (in meters)
        self.ring_depth_samples = 10  # Number of depth samples to check

        #Color detection
        self.pure_color_hsv_histograms = {
            "red": generate_hsv_color_hist((0, 0, 255)),
            "green": generate_hsv_color_hist((0, 255, 0)),
            "blue": generate_hsv_color_hist((255, 0, 0)),
        }
        #for name, hist in self.pure_color_hsv_histograms.items():
        #    print(f"{name}: shape={hist.shape}, dtype={hist.dtype}, sum={np.sum(hist):.4f}")

        #Ring publishing
        #self.rings = []
        #self.face_groups = []
        #self.detected_faces_sent = set()


    def get_ring_color(self, mask_ring, color_buffer_hsv):
        if cv2.countNonZero(mask_ring) == 0:
            self.get_logger().warn("Empty ring mask, skipping color detection.")
            return None, 0.0

        # Visualize masked region
        #bgr_image = cv2.cvtColor(color_buffer_hsv, cv2.COLOR_HSV2BGR)
        #bgr_image_masked = cv2.bitwise_and(bgr_image, bgr_image, mask=mask_ring)
        #black_pixels = np.all(bgr_image_masked == [0, 0, 0], axis=-1)
        #bgr_image_masked[black_pixels] = [255, 255, 255]
        #cv2.imshow("Ring Region (white background)", bgr_image_masked)
        #cv2.waitKey(1)

        #black is the edge case because it does not depend on HUE
        v_channel = color_buffer_hsv[:, :, 2]
        ring_v_values = v_channel[mask_ring == 255]
        intensity = np.mean(ring_v_values)

        s_channel = color_buffer_hsv[:, :, 1]
        ring_s_values = s_channel[mask_ring == 255]
        saturation = np.mean(ring_s_values)

        #color strongly suggests black
        intensity_val = intensity / 255
        print(intensity_val)
        if (intensity_val < 0.45):
            return "black", 1.0

        

        # Histogram of masked pixels
        hue_hist = cv2.calcHist([color_buffer_hsv], [0], mask_ring, [180], [0, 180])
        hue_hist = hue_hist.astype(np.float32)
        hue_hist = cv2.normalize(hue_hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hue_hist = hue_hist.reshape(-1, 1)


        # Plot histogram
        #plt.plot(hue_hist)
        #plt.title("Hue Histogram for Ring")
        #plt.xlabel("Hue")
        #plt.ylabel("Normalized Frequency")
        #plt.show()

        # Classify by comparing with reference histograms
        scores = {}
        for color_name, ref_hist in self.pure_color_hsv_histograms.items():
            score = cv2.compareHist(hue_hist, ref_hist, cv2.HISTCMP_INTERSECT)
            scores[color_name] = score

        total = sum(scores.values())
        if total == 0:
            print("got no scores")
            return None, 0.0

        probabilities = {color: score / total for color, score in scores.items()}
        top_color = max(probabilities, key=probabilities.get)

        # Plot probability bar chart
        #labels = list(probabilities.keys())
        #values = list(probabilities.values())
        #plt.figure(figsize=(8, 4))
        #bars = plt.bar(labels, values, color=labels)
        #plt.title("Color probabilities")
        #plt.ylabel("Probability")
        #plt.ylim(0, 1.0)
        #for bar, value in zip(bars, values):
        #    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
        #        f"{value:.2f}", ha='center', va='bottom')
        #plt.grid(axis='y', linestyle='--', alpha=0.5)
        #plt.tight_layout()
        #plt.show()

        return top_color, probabilities[top_color]




    def image_callback(self, data):
        if self.latest_depth is None:
            self.get_logger().info("No depth image received yet, skipping ring detection")
            return

        #self.get_logger().info("Processing new image for ring detection")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            depth_image = self.latest_depth.copy()
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting images: {e}")
            return

        # Tranform image to grayscale
        # gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # extract the top half of the image
        # gray = gray[0:int(gray.shape[0]/2),:]
        
        # Threshold the image
        # gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                                cv2.THRESH_BINARY_INV, 15, 9)

            # Convert to HSV color space for better color segmentation
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Separate channels
        h, saturation, v = cv2.split(hsv_image)

        #sky filtering
        sky_mask = depth_image == np.inf  # Keep only pixels with non-sky depth (tune threshold if needed)
        masked_sat= saturation.copy()
        masked_sat[sky_mask] = 0
        saturation = masked_sat
        
        # Create a mask of highly saturated areas (likely rings)
        # White/gray rods will have low saturation
        saturation_threshold = 70  # Adjust based on your specific scenario
        # color_mask = cv2.threshold(saturation, saturation_threshold, 255, cv2.THRESH_BINARY)[1]
        color_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Method 2: More controlled morphology
        # Skip the dilation and use a small kernel for better precision
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Close small gaps in the rings without thickening too much
        processed_mask = cv2.dilate(color_mask, None, iterations=1)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Find edges of the processed mask
        thresh = cv2.Canny(processed_mask, 50, 150)

        # Extract contours - use the edge image for thinner contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


        
        # Show binary image
        cv2.imshow("Binary Image", thresh)
        cv2.waitKey(1)

        

        
        # # Fit elipses to all extracted contours
        # # After extracting contours but before ellipse fitting
        filtered_contours = []
        for cnt in contours:
            # Skip contours that are too small
            if len(cnt) < 15:
                continue
            
            # Check if contour is ellipse-like using circularity
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.6:  # Adjust threshold as needed (0.6-0.8 is good for ellipses)
                continue
            
            # Add to filtered contours
            filtered_contours.append(cnt)

        # Use filtered contours instead of all contours
        contours = filtered_contours

        # Draw contours on a copy of the grayscale image
        contour_image = cv2.cvtColor(saturation, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
        cv2.imshow("Detected contours", contour_image)
        cv2.waitKey(1)

        # Fit ellipses to the filtered contours
        ellipses = []
        for cnt in contours:
            if cnt.shape[0] >= 15:
                ellipse = cv2.fitEllipse(cnt)
                ellipses.append(ellipse)

        # Find pairs of concentric ellipses (potential rings)
        ring_candidates = []
        for i in range(len(ellipses)):
            for j in range(i + 1, len(ellipses)):
                e1 = ellipses[i]
                e2 = ellipses[j]
                
                # Calculate center distance
                center_dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                if center_dist > self.max_center_distance:
                    continue
                
                # Check angle difference
                angle_diff = np.abs(e1[2] - e2[2])
                if angle_diff > self.max_angle_diff:
                    continue
                
                # Determine which ellipse is larger
                e1_area = np.pi * e1[1][0] * e1[1][1] / 4
                e2_area = np.pi * e2[1][0] * e2[1][1] / 4
                
                if e1_area > e2_area:
                    larger = e1
                    smaller = e2
                else:
                    larger = e2
                    smaller = e1
                
                # Check if one ellipse contains the other
                l_major, l_minor = max(larger[1]), min(larger[1])
                s_major, s_minor = max(smaller[1]), min(smaller[1])
                
                # # Ring width check - difference between the radii
                ring_width = (l_minor - s_minor) / 2
                if ring_width < self.min_ring_width or ring_width > self.max_ring_width:
                    continue

                if l_minor < self.min_circle_height or s_minor < self.min_circle_height:
                    continue
                
                # Calculate aspect ratio of the ring
                aspect_ratio_l = l_major / l_minor
                aspect_ratio_s = s_major / s_minor
                if aspect_ratio_l > 1.5 or aspect_ratio_s > 1.5:  # Reject highly elliptical rings
                    continue
                border_major = (l_major - s_major) / 2
                border_minor = (l_minor - s_minor) / 2
                border_diff = np.abs(border_major - border_minor)
                if border_diff > 5:
                    continue
                
                
                #print("\033[91mRing detected\033[0m")  # Red text in terminal
                
                # Check if ring is 3D using depth information
                if self.is_3d_ring(depth_image, larger[0], l_major, l_minor, smaller[0], s_major, s_minor):
                    #print("\033[91mRing detected\033[0m")  # Red text in terminal
                    ring_candidates.append((larger, smaller))

                # Plot the rings on the image
        for c in ring_candidates:
            #elipsis
            bigger_ellipse = c[0]
            smaller_ellipse = c[1]

            # Make ring mask
            mask_e1 = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask_e1, bigger_ellipse, 255, thickness=-1)
            mask_e2 = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask_e2, smaller_ellipse, 255, thickness=-1)

            mask_ring = cv2.subtract(mask_e1, mask_e2)
            valid_depth_mask = np.isfinite(depth_image).astype(np.uint8) * 255
            mask_ring = cv2.bitwise_and(mask_ring, valid_depth_mask)

            #getting color
            color, p = self.get_ring_color(mask_ring, hsv_image)

            #getting depth
            masked_depth = depth_image[mask_ring > 0]
            valid_depths = masked_depth[(masked_depth > 0) & np.isfinite(masked_depth)]

            if valid_depths.size > 0:
                median_depth = np.median(valid_depths)
            else:
                return

            if color is not None:
                if self.latest_pointcloud is None:
                    self.get_logger().warn("No point cloud received yet, skipping 3D conversion.")
                    return

                try:
                    # Read point cloud as numpy array (H, W, 3)
                    pc_array = pc2.read_points_numpy(
                        self.latest_pointcloud, field_names=("x", "y", "z")
                    ).reshape((self.latest_pointcloud.height, self.latest_pointcloud.width, 3))

                    # Get (u, v) indices where mask_ring is 255
                    ys, xs = np.where(mask_ring == 255)

                    # Extract 3D points from the point cloud at those indices
                    points_3d = pc_array[ys, xs]  # shape: (N, 3)

                    # Filter out NaN or zero-length vectors
                    valid_points = points_3d[
                        np.isfinite(points_3d).all(axis=1) & (np.linalg.norm(points_3d, axis=1) > 0.05)
                    ]

                    if valid_points.shape[0] < 3:
                        self.get_logger().warn("Not enough valid 3D points for averaging.")
                        return

                    avg_3d = np.mean(valid_points, axis=0)

                    # Transform to map frame
                    stamped = PointStamped()
                    stamped.header.stamp = self.get_clock().now().to_msg()
                    stamped.header.frame_id = self.latest_pointcloud.header.frame_id
                    stamped.point.x = float(avg_3d[0])
                    stamped.point.y = float(avg_3d[1])
                    stamped.point.z = float(avg_3d[2])

                    transform = self.tf_buffer.lookup_transform(
                        target_frame="map",
                        source_frame=stamped.header.frame_id,
                        time=rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.5)
                    )
                    transformed = tf2_geometry_msgs.do_transform_point(stamped, transform)

                    map_point = np.array([transformed.point.x, transformed.point.y, transformed.point.z])

                    #self.get_logger().warn(f"RING AT {map_point}")
                    self.add_ring_to_group(map_point, color)

                except Exception as e:
                    self.get_logger().warn(f"Failed to get averaged 3D point for ring: {e}")

            # drawing the ellipses on the image
            cv2.ellipse(cv_image, bigger_ellipse, (0, 255, 0), 2)
            cv2.ellipse(cv_image, smaller_ellipse, (0, 255, 0), 2)

            # Get a bounding box, around the first ellipse ('average' of both elipsis)
            #size = (e1[1][0]+e1[1][1])/2
            #center = (e1[0][1], e1[0][0])

            #x1 = int(center[0] - size / 2)
            #x2 = int(center[0] + size / 2)
            #x_min = x1 if x1>0 else 0
            #x_max = x2 if x2<cv_image.shape[0] else cv_image.shape[0]

            #y1 = int(center[1] - size / 2)
            #y2 = int(center[1] + size / 2)
            #y_min = y1 if y1 > 0 else 0
            #y_max = y2 if y2 < cv_image.shape[1] else cv_image.shape[1]

        if len(ring_candidates) > 0:
            cv2.imshow("Detected rings",cv_image)
            cv2.waitKey(1)




        

    def is_3d_ring(self, depth_image, outer_center, outer_major, outer_minor, 
                   inner_center, inner_major, inner_minor):
        """
        Validate if a ring candidate is actually a 3D ring using depth information.
        A 3D ring should have a depth discontinuity in the middle compared to the ring itself.
        """
        # Get center coordinates
        cx, cy = int(outer_center[0]), int(outer_center[1])
        
        # Check if center is within image bounds
        if cx >= depth_image.shape[1] or cy >= depth_image.shape[0] or cx < 0 or cy < 0:
            return False
        
        # Calculate the radius of the inner hole
        inner_radius = min(inner_major, inner_minor) / 2
        
        # Calculate the outer radius 
        outer_radius = min(outer_major, outer_minor) / 2
        
        # Sample depth values in the center, on the ring, and outside
        center_depths = []
        ring_depths = []
        
        # Sample points in a grid pattern
        for angle in np.linspace(0, 2*np.pi, self.ring_depth_samples, endpoint=False):
            # Points inside the hole (center)
            r_center = inner_radius * 0.5
            x_center = int(cx + r_center * np.cos(angle))
            y_center = int(cy + r_center * np.sin(angle))
            
            # Create a new visualization image to show sample points
            if not hasattr(self, 'sample_points_img'):
                self.sample_points_img = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
            else:
                # Clear the image for new points
                self.sample_points_img = np.zeros_like(self.sample_points_img)

            # Draw the center point in red
            cv2.circle(self.sample_points_img, (x_center, y_center), 3, (0, 0, 255), -1)
            cv2.imshow("Depth Sample Points", self.sample_points_img)
            cv2.waitKey(1)
            
            if 0 <= x_center < depth_image.shape[1] and 0 <= y_center < depth_image.shape[0]:
                depth = depth_image[y_center, x_center]
                if depth > 0:
                    center_depths.append(depth)
            
            # Points on the ring
            r_ring = (inner_radius + outer_radius) / 2
            x_ring = int(cx + r_ring * np.cos(angle))
            y_ring = int(cy + r_ring * np.sin(angle))
            
            if 0 <= x_ring < depth_image.shape[1] and 0 <= y_ring < depth_image.shape[0]:
                depth = depth_image[y_ring, x_ring]
                if depth > 0 and not np.isinf(depth):
                    ring_depths.append(depth)
        
        #print(f"\033[92mCenter depths: {center_depths}\033[0m")
        #print(f"\033[92mRing depths: {ring_depths}\033[0m")
        
        # Check if we have enough valid depth samples
        if len(center_depths) < 1 or len(ring_depths) < 1:
            return False
        
        # Calculate median depths to reduce noise impact
        center_depth = np.median(center_depths)
        ring_depth = np.median(ring_depths)
        
        # Calculate depth difference
        depth_diff = center_depth - ring_depth
        
        # If the center is significantly farther than the ring, it's likely a 3D ring
        if depth_diff > self.depth_threshold:
            return True
            
        return False

    def depth_callback(self, data):
        try:
            # Convert depth image to meters
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            self.latest_depth = depth_image
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting depth image: {e}")
            return


        
        cv2.imshow("Depth window", depth_image)
        cv2.waitKey(1)

    def deproject_pixel_to_point(self, u, v, depth, fx, fy, cx, cy):
        """Convert pixel (u,v) + depth to 3D point in camera coordinates."""
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        return np.array([X, Y, Z])

    def pc_callback(self, msg):
        self.latest_pointcloud = msg


    def add_ring_to_group(self, position, color, threshold=0.5):
        for group in self.ring_groups:
            if np.linalg.norm(group['position'] - position) < threshold:
                group['positions'].append(position)
                group['colors'].append(color)
                return
        self.ring_groups.append({'positions': [position], 'colors': [color], 'position': position})

    def publish_new_rings(self):
        for group in self.ring_groups:
            if len(group['positions']) < 3:
                continue

            avg_pos = np.mean(group['positions'], axis=0)
            key = tuple(np.round(avg_pos, 2))
            if key in self.detected_rings_sent:
                continue

            # Majority color vote
            color_counts = {}
            for c in group['colors']:
                color_counts[c] = color_counts.get(c, 0) + 1
            majority_color = max(color_counts, key=color_counts.get)

            try:
                msg = DetectedRing()
                msg.position.header.stamp = self.get_clock().now().to_msg()
                msg.position.header.frame_id = "map"  # ✅ Position is already in map frame
                msg.position.point.x = float(avg_pos[0])
                msg.position.point.y = float(avg_pos[1])
                msg.position.point.z = float(avg_pos[2])
                msg.color = majority_color

                self.detected_ring_pub.publish(msg)
                self.detected_rings_sent.add(key)
                self.get_logger().info(f"🔔 Published ring at {avg_pos.round(2)} with color {majority_color}")

            except Exception as e:
                self.get_logger().warn(f"Failed to publish ring: {e}")


def main():
    rclpy.init(args=None)
    ring_detector = RingDetector()
    
    try:
        rclpy.spin(ring_detector)
    except KeyboardInterrupt:
        pass
    finally:
        ring_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


    
