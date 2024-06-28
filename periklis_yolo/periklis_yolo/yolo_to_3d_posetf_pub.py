import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import multiprocessing
import signal
import argparse
import yaml
import os

from ember_detection_interfaces.msg import EmberBoundingBox3D, EmberBoundingBox3DArray

from periklis_yolo.utils import *
from periklis_yolo.visualization.o3d_detect_viz import Open3DDetectVisualizer
from periklis_yolo.visualization.cv2_detect_viz import OpenCV2DetectVisualizer

def o3d_vis_worker(o3d_vis_input_queue):

    def sigint_handler(sig, frame):
        o3d_vis_input_queue.put(None)
        exit(0)

    # sigint exit
    signal.signal(signal.SIGINT, sigint_handler)

    visualizer = Open3DDetectVisualizer()
    while True:
        data = o3d_vis_input_queue.get()
        if data is None:
            break
        visualizer.reset()
        bboxes = data['bboxes']
        for bbox in bboxes:
            visualizer.draw_bbox(bbox)
        pc2_points_64 = pc2_msg_to_numpy(data['pointcloud_msg'])
        visualizer.draw_pointcloud(pc2_points_64, data['transformation_matrix'])
        visualizer.render()

def cv2_vis_worker(cv2_vis_input_queue):

    def sigint_handler(sig, frame):
        cv2_vis_input_queue.put(None)
        exit(0)

    # sigint exit
    signal.signal(signal.SIGINT, sigint_handler)

    visualizer = OpenCV2DetectVisualizer()
    while True:
        data = cv2_vis_input_queue.get()
        if data is None:
            break
        bgr_image = data['bgr_image']
        bboxes_data = data['bboxes']
        for bbox_data in bboxes_data:
            bbox_label = bbox_data['label']
            bbox_points = bbox_data['points']
            visualizer.draw_cv2_bounding_box(bbox_points, bbox_label, bgr_image)
        cv2.imshow(visualizer.window_name, bgr_image)
        cv2.waitKey(1)

class YoloTo3DPoseTransformPub(Node):
    def __init__(self, config_file, model_file,
                 o3d_vis_input_queue, cv2_vis_input_queue):
        super().__init__('yolo_to_3d_pose_transform')

        self.load_config(config_file)

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
        
        self.detection_pub = self.create_publisher(EmberBoundingBox3DArray, 'ember_detection/ember_bounding_boxes', 10)
        
        self.pose_subscription
        self.current_pose = None

        self.model = YOLO(model_file) # default model file is yolov8n.engine

        self.o3d_vis_input_queue = o3d_vis_input_queue
        self.cv2_vis_input_queue = cv2_vis_input_queue

    def load_config(self, config_file):
        if config_file is not None:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                try:
                    self.bgr_resize_factor = config['bgr_resize_factor']
                    self.zed_ros2_wrapper_downscale_factor = config['zed_ros2_wrapper_downscale_factor']
                except KeyError as e:
                    self.get_logger().error(f'Error loading config file: {e}')
                    exit(1)

    def pose_callback(self, msg):
        self.current_pose = msg

    def main_callback(self, msg_image, msg_camera_info, msg_depth, msg_pointcloud):

        fx, fy = msg_camera_info.k[0], msg_camera_info.k[4]
        cx, cy = msg_camera_info.k[2], msg_camera_info.k[5]

        bgr_image = CvBridge().imgmsg_to_cv2(msg_image, desired_encoding='bgr8')
        depth_image = CvBridge().imgmsg_to_cv2(msg_depth, desired_encoding='passthrough')

        # TODO read this settings from config file
        bgr_detection_resize_factor = 4
        zed_ros2_wrapper_downscale_factor = 8
        bgr_resized_to_depth_ratio = int(zed_ros2_wrapper_downscale_factor / bgr_detection_resize_factor)

        fx /= zed_ros2_wrapper_downscale_factor
        fy /= zed_ros2_wrapper_downscale_factor
        cx /= zed_ros2_wrapper_downscale_factor
        cy /= zed_ros2_wrapper_downscale_factor

        bgr_resized = cv2.resize(bgr_image, (bgr_image.shape[1] // bgr_detection_resize_factor, bgr_image.shape[0] // bgr_detection_resize_factor))
        results = self.model(bgr_resized)[0]

        transformation_matrix = None
        if self.current_pose is not None:
            transformation_matrix = pose_msg_to_transform_matrix(self.current_pose)

        # data 3d visualization
        o3d_vis_input_data = {
            'bboxes': [],
            'pointcloud_msg': None,
            'transformation_matrix': transformation_matrix
        }

        # data 2d visualization
        cv2_vis_input_data = {
            'bgr_image': bgr_resized,
            'bboxes': []
        }

        ember_bbox_array = EmberBoundingBox3DArray()

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

                confidence = boxes[i].conf[0].item()
                class_id = int(boxes[i].cls[0].item())
                label = f'{self.model.names[class_id]}: {confidence:.2f}'

                cv2_vis_input_data['bboxes'].append({
                    'label': label,
                    'points': (u1, v1, u2, v2)
                })

                o3d_vis_input_data['bboxes'].append(bbox3d_points)

                ember_bbox = build_ember_bbox(bbox3d_points)
                ember_bbox_array.boxes.append(ember_bbox)

        o3d_vis_input_data['pointcloud_msg'] = msg_pointcloud

        ember_bbox_array.header = Header()
        ember_bbox_array.header.stamp = msg_depth.header.stamp
        self.detection_pub.publish(ember_bbox_array)
        
        self.o3d_vis_input_queue.put(o3d_vis_input_data)
        self.cv2_vis_input_queue.put(cv2_vis_input_data)

def signal_handler(sig, frame, node, process_1, queue_1, process_2, queue_2):
    print('Exiting via signal handler...')
    queue_1.put(None)
    queue_2.put(None)
    process_1.terminate()
    process_1.join()
    process_2.terminate()
    process_2.join()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

def main(args=None):

    # argument parsing
    parser = argparse.ArgumentParser(description='Yolo detection to 3D boundingbox with pose transform Publisher')
    parser.add_argument('--config_file', type=str, help='Path to config file',
                        default='src/periklis_yolo/config/yolo_to_3d_pose_transform.yaml')
    parser.add_argument('--model_file', type=str, help='Path to model file', 
                        default='src/periklis_yolo/models/yolov8n.engine')
    parsed_args = parser.parse_args()

    print(f'Arguments: {parsed_args}')

    o3d_vis_input_queue = multiprocessing.Queue()
    cv2_vis_input_queue = multiprocessing.Queue()

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, node, o3d_vis_process, o3d_vis_input_queue, cv2_vis_process, cv2_vis_input_queue))

    o3d_vis_process = multiprocessing.Process(target=o3d_vis_worker, args=(o3d_vis_input_queue,))
    o3d_vis_process.start()

    cv2_vis_process = multiprocessing.Process(target=cv2_vis_worker, args=(cv2_vis_input_queue,))
    cv2_vis_process.start()

    rclpy.init(args=args)
    node = YoloTo3DPoseTransformPub(parsed_args.config_file, parsed_args.model_file,
        o3d_vis_input_queue, cv2_vis_input_queue)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print('Exiting via finally...')
        o3d_vis_input_queue.put(None)
        cv2_vis_input_queue.put(None)
        o3d_vis_process.terminate()
        o3d_vis_process.join()
        cv2_vis_process.terminate()
        cv2_vis_process.join()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        

if __name__ == '__main__':
    main()
