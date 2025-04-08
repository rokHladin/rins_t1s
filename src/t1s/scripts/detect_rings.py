#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

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
        
        # Subscribe to the image and/or depth topic
        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
        
        # Publisher for ring detections
        self.ring_pub = self.create_publisher(Marker, "/detected_ring", 10)
        self.ring_info_pub = self.create_publisher(String, "/ring_info", 10)

        # Object we use for transforming between coordinate frames
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Create windows for visualization
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        
        # Parameters for ring detection
        self.min_contour_points = 20
        self.max_center_distance = 5.0
        self.max_angle_diff = 7.0
        self.min_ring_width = 2
        self.max_ring_width = 50
        self.min_circle_height = 4
        
        # Parameters for 3D validation
        self.depth_threshold = 0.1  # Expected depth difference for 3D rings (in meters)
        self.ring_depth_samples = 3  # Number of depth samples to check

    def image_callback(self, data):
        if self.latest_depth is None:
            self.get_logger().info("No depth image received yet, skipping ring detection")
            return

        self.get_logger().info("Processing new image for ring detection")

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
                    print("\033[91mRing detected\033[0m")  # Red text in terminal
                    ring_candidates.append((larger, smaller))

                # Plot the rings on the image
        for c in ring_candidates:

            # the centers of the ellipses
            e1 = c[0]
            e2 = c[1]

            # drawing the ellipses on the image
            cv2.ellipse(cv_image, e1, (0, 255, 0), 2)
            cv2.ellipse(cv_image, e2, (0, 255, 0), 2)

            # Get a bounding box, around the first ellipse ('average' of both elipsis)
            size = (e1[1][0]+e1[1][1])/2
            center = (e1[0][1], e1[0][0])

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1>0 else 0
            x_max = x2 if x2<cv_image.shape[0] else cv_image.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < cv_image.shape[1] else cv_image.shape[1]

        if len(ring_candidates)>0:
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
        
        print(f"\033[92mCenter depths: {center_depths}\033[0m")
        print(f"\033[92mRing depths: {ring_depths}\033[0m")
        
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

        # Process the depth image for visualization
        depth_viz = depth_image.copy()
        depth_viz[depth_viz == np.inf] = 0
        depth_viz[depth_viz > 10] = 10  # Cap maximum distance for visualization
        
        # Normalize for display
        depth_viz_norm = depth_viz / np.max(depth_viz) if np.max(depth_viz) > 0 else depth_viz
        depth_viz_display = (depth_viz_norm * 255).astype(np.uint8)
        
        # Apply a colormap for better visualization
        depth_colormap = cv2.applyColorMap(depth_viz_display, cv2.COLORMAP_JET)
        
        cv2.imshow("Depth window", depth_colormap)
        cv2.waitKey(1)

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