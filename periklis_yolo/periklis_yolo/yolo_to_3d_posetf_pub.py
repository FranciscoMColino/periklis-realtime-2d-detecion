import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
import sensor_msgs_py.point_cloud2 as pc2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import open3d as o3d

from periklis_yolo.utils import *
from periklis_yolo.visualization.o3d_detect_viz import Open3DDetectVisualizer

class YoloTo3DPoseTransformPub(Node):
    def __init__(self):
        super().__init__('yolo_to_3d_pose_transform')
        self.image_sub = Subscriber(self, Image, '/zed/zed_node/left_original/image_rect_color')
        self.camera_info_sub = Subscriber(self, CameraInfo, '/zed/zed_node/left_original/camera_info')
        self.depth_sub = Subscriber(self, Image, '/zed/zed_node/depth/depth_registered')
        self.pointcloud_sub = Subscriber(self, PointCloud2, '/zed/zed_node/point_cloud/cloud_registered')

        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.camera_info_sub, self.depth_sub, self.pointcloud_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.main_callback)

        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/zed/zed_node/pose',
            self.pose_callback,
            10)
        
        self.pose_subscription
        self.current_pose = None

        self.visualizer = Open3DDetectVisualizer()

        self.model = YOLO('yolov8n.engine')

    def draw_cv2_bounding_box(self, box, bbox, bgr_image):
        u1, v1, u2, v2 = bbox
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        label = f'{self.model.names[class_id]}: {confidence:.2f}'
        color = (0, 255, 0)
        cv2.rectangle(bgr_image, (u1, v1), (u2, v2), color, 2)
        cv2.putText(bgr_image, label, (u1, v1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    def pose_callback(self, msg):
        print('Received pose')
        self.current_pose = msg

    def main_callback(self, msg_image, msg_camera_info, msg_depth, msg_pointcloud):
        
        self.visualizer.reset()

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
        results = self.model(bgr_resized)[0]

        transformation_matrix = None
        if self.current_pose is not None:
            transformation_matrix = pose_msg_to_transform_matrix(self.current_pose)

        for result in results:
            boxes = result.boxes

            for i in range(len(boxes)):

                # if class is not person, skip
                if boxes[i].cls[0].item() != 0:
                    continue

                box = np.array(boxes[i].xyxy[0].tolist())

                u1, v1, u2, v2 = map(int, box)

                parent_point_1, parent_point_2 = compute_parents_from_bbox(np.array([u1, v1, u2, v2]), bgr_resized_to_depth_ratio, fx, fy, cx, cy, depth_image)

                if transformation_matrix is not None:
                    parent_point_1 = apply_transformation(parent_point_1, transformation_matrix)
                    parent_point_2 = apply_transformation(parent_point_2, transformation_matrix)

                bbox3d_points = compute_3d_bbox_from_parents_2(parent_point_1, parent_point_2)

                self.draw_cv2_bounding_box(boxes[i], (u1, v1, u2, v2), bgr_resized)
                self.visualizer.draw_bbox(bbox3d_points)

        cv2.imshow('YOLOv8 Detection', bgr_resized)
        cv2.waitKey(1)

        pc2_points_64 = pc2_msg_to_numpy(msg_pointcloud)
        self.visualizer.draw_pointcloud(pc2_points_64, transformation_matrix)
        self.visualizer.render()

def main(args=None):
    rclpy.init(args=args)
    node = YoloTo3DPoseTransformPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
