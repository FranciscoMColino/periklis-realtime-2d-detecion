import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from ultralytics import YOLO
import open3d as o3d

from ember_detection_interfaces.msg import EmberBoundingBox3D, EmberBoundingBox3DArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header, String, UInt32

class YoloTo3DPub(Node):
    def __init__(self):
        super().__init__('yolo_to_3d')
        self.image_sub = Subscriber(self, Image, '/zed/zed_node/left_original/image_rect_color')
        self.camera_info_sub = Subscriber(self, CameraInfo, '/zed/zed_node/left_original/camera_info')
        self.depth_sub = Subscriber(self, Image, '/zed/zed_node/depth/depth_registered')
        self.bbox_pub = self.create_publisher(EmberBoundingBox3DArray, 'ember_detection/ember_bounding_boxes', 10)

        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.camera_info_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.model = YOLO('yolov8n.engine')

    def compute_points_from_bbox(self, bbox, resize_factor, fx, fy, cx, cy, depth_image):
        u1, v1, u2, v2 = bbox // resize_factor

        # check v1, v2, u1, u2 bounds
        v1 = max(0, v1)
        v1 = min(depth_image.shape[0] - 1, v1)
        v2 = max(0, v2)
        v2 = min(depth_image.shape[0] - 1, v2)
        u1 = max(0, u1)
        u1 = min(depth_image.shape[1] - 1, u1)
        u2 = max(0, u2)
        u2 = min(depth_image.shape[1] - 1, u2)

        depth_values_bb = depth_image[v1:v2, u1:u2]
        depth_values_bb = depth_values_bb[np.isfinite(depth_values_bb)]
        if len(depth_values_bb) == 0:
            print('No finite depth values in the bounding box')
            return None
        min_depth = np.min(depth_values_bb)

        z1 = min_depth
        x1 = ((u1 - cx) * z1) / fx
        y1 = ((v1 - cy) * z1) / fy

        z_h = min_depth
        x_h = ((u2 - cx) * z_h) / fx
        #y_h = ((v2 - cy) * z_h) / fy

        z2 = min_depth + np.abs(x_h - x1)
        x2 = ((u2 - cx) * z2) / fx
        y2 = ((v2 - cy) * z2) / fy

        x1, y1, z1 = z1, -x1, -y1
        x2, y2, z2 = z2, -x2, -y2

        points = np.array([
                    [x1, y1, z1],
                    [x1, y1, z2],
                    [x1, y2, z1],
                    [x1, y2, z2],
                    [x2, y1, z1],
                    [x2, y1, z2],
                    [x2, y2, z1],
                    [x2, y2, z2]
                ])

        return points
    
    def build_ember_bbox(self, points):
        ember_bbox = EmberBoundingBox3D()
        for point in points:
            ember_bbox.points.append(Point(x=point[0], y=point[1], z=point[2]))
        ember_bbox.det_label = String(data='person')
        ember_bbox.points_count = UInt32(data=len(points))
        return ember_bbox

    def draw_cv2_bounding_box(self, box, bbox, bgr_image):
        u1, v1, u2, v2 = bbox
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        label = f'{self.model.names[class_id]}: {confidence:.2f}'
        color = (0, 255, 0)
        cv2.rectangle(bgr_image, (u1, v1), (u2, v2), color, 2)
        cv2.putText(bgr_image, label, (u1, v1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        

    def callback(self, msg_image, msg_camera_info, msg_depth):
        # log timestamp
        self.get_logger().info(f"ImageDepthSyncVizSubscriber callback: {msg_image.header.stamp}")

        fx, fy = msg_camera_info.k[0], msg_camera_info.k[4]
        cx, cy = msg_camera_info.k[2], msg_camera_info.k[5]

        bgr_image = CvBridge().imgmsg_to_cv2(msg_image, desired_encoding='bgr8')
        depth_image = CvBridge().imgmsg_to_cv2(msg_depth, desired_encoding='passthrough')

        # TODO read this settings from config file
        bgr_detection_resize_factor = 4
        zed_ros2_wrapper_downscale_factor = 8
        bgr_resized_to_depth_ratio = 2

        fx /= zed_ros2_wrapper_downscale_factor
        fy /= zed_ros2_wrapper_downscale_factor
        cx /= zed_ros2_wrapper_downscale_factor
        cy /= zed_ros2_wrapper_downscale_factor

        bgr_resized = cv2.resize(bgr_image, (bgr_image.shape[1] // bgr_detection_resize_factor, bgr_image.shape[0] // bgr_detection_resize_factor))

        ember_bbox_array = EmberBoundingBox3DArray()

        results = self.model(bgr_resized)[0]

        for result in results:
            boxes = result.boxes
            # masks = result.masks TODO
            for i in range(len(boxes)):

                # if class is not person, skip
                if boxes[i].cls[0].item() != 0:
                    continue

                u1, v1, u2, v2 = map(int, boxes[i].xyxy[0].tolist())

                points = self.compute_points_from_bbox(np.array([u1, v1, u2, v2]), bgr_resized_to_depth_ratio, fx, fy, cx, cy, depth_image)

                ember_bbox = self.build_ember_bbox(points)
                ember_bbox_array.boxes.append(ember_bbox)
                self.draw_cv2_bounding_box(boxes[i], (u1, v1, u2, v2), bgr_resized)
        
        ember_bbox_array.header = Header()
        ember_bbox_array.header.stamp = msg_depth.header.stamp
        
        self.bbox_pub.publish(ember_bbox_array)
  
        cv2.imshow('YOLOv8 Detection', bgr_resized)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloTo3DPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
