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

class YoloMask3DBbox(Node):
    def __init__(self):
        super().__init__('yolo_to_3d')
        self.image_sub = Subscriber(self, Image, '/zed/zed_node/left_original/image_rect_color')
        self.camera_info_sub = Subscriber(self, CameraInfo, '/zed/zed_node/left_original/camera_info')
        self.depth_sub = Subscriber(self, Image, '/zed/zed_node/depth/depth_registered')
        self.pointcloud_sub = Subscriber(self, PointCloud2, '/zed/zed_node/point_cloud/cloud_registered')

        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.camera_info_sub, self.depth_sub, self.pointcloud_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('Open3D', width=640, height=480)
        self.setup_visualizer()

        self.model = YOLO('yolov8n.engine')

    def setup_visualizer(self):
        # Add 8 points to initiate the visualizer's bounding box
        points = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [10, 0, 0],
            [10, 0, 1],
            [10, 1, 0],
            [10, 1, 1]
        ])

        points *= 4

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        self.vis.add_geometry(pcd, reset_bounding_box=True)

        view_control = self.vis.get_view_control()
        view_control.rotate(0, -525)
        view_control.rotate(500, 0)

        # points thinner and lines thicker
        self.vis.get_render_option().point_size = 2.0
        self.vis.get_render_option().line_width = 10.0

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

        percentile = 15
        percentile_depth_values = np.percentile(depth_values_bb, percentile)

        z1 =  percentile_depth_values * 1.15

        if not np.isfinite(z1):
            # use median depth value around the point
            median_filter_radius = 3
            z1 = np.nanmedian(depth_image[max(0, v1 - median_filter_radius):min(depth_image.shape[0] - 1, v1 + median_filter_radius), max(0, u1 - median_filter_radius):min(depth_image.shape[1] - 1, u1 + median_filter_radius)])


        x1 = ((u1 - cx) * z1) / fx
        y1 = ((v1 - cy) * z1) / fy

        x1, y1, z1 = z1, -x1, -y1

        z2 = min_depth

        if not np.isfinite(z2):
            # use median depth value around the point
            median_filter_radius = 3
            z2 = np.nanmedian(depth_image[max(0, v2 - median_filter_radius):min(depth_image.shape[0] - 1, v2 + median_filter_radius), max(0, u2 - median_filter_radius):min(depth_image.shape[1] - 1, u2 + median_filter_radius)])

        x2 = ((u2 - cx) * z2) / fx
        y2 = ((v2 - cy) * z2) / fy

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

    def draw_cv2_bounding_box(self, box, bbox, bgr_image):
        u1, v1, u2, v2 = bbox
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        label = f'{self.model.names[class_id]}: {confidence:.2f}'
        color = (0, 255, 0)
        cv2.rectangle(bgr_image, (u1, v1), (u2, v2), color, 2)
        cv2.putText(bgr_image, label, (u1, v1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        

    def callback(self, msg_image, msg_camera_info, msg_depth, msg_pointcloud):
        
        self.vis.clear_geometries()
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5), reset_bounding_box=False)

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

        for result in results:
            boxes = result.boxes
            # masks = result.masks TODO
            for i in range(len(boxes)):

                # if class is not person, skip
                if boxes[i].cls[0].item() != 0:
                    continue

                u1, v1, u2, v2 = map(int, boxes[i].xyxy[0].tolist())

                points = self.compute_points_from_bbox(np.array([u1, v1, u2, v2]), bgr_resized_to_depth_ratio, fx, fy, cx, cy, depth_image)
                self.draw_cv2_bounding_box(boxes[i], (u1, v1, u2, v2), bgr_resized)
        
                # Display computed bounding box o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                self.vis.add_geometry(pcd, reset_bounding_box=False)

                bbox = pcd.get_oriented_bounding_box()
                bbox.color = (1, 0, 0)
                self.vis.add_geometry(bbox, reset_bounding_box=False)

        cv2.imshow('YOLOv8 Detection', bgr_resized)
        cv2.waitKey(1)

        pc2_points = pc2.read_points_numpy(msg_pointcloud, field_names=('x', 'y', 'z'), skip_nans=True)
        pc2_points_64 = pc2_points.astype(np.float64)
        valid_idx = np.isfinite(pc2_points_64[:, 0]) & np.isfinite(pc2_points_64[:, 1]) & np.isfinite(pc2_points_64[:, 2])
        pc2_points_64 = pc2_points_64[valid_idx]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc2_points_64)
        self.vis.add_geometry(pcd, reset_bounding_box=False)

        self.vis.poll_events()
        self.vis.update_renderer()

def main(args=None):
    rclpy.init(args=args)
    node = YoloMask3DBbox()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
