import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from ultralytics import YOLO
import open3d as o3d

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
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
from std_msgs.msg import Header

class ImageDepthSyncVizSubscriber(Node):
    def __init__(self):
        super().__init__('image_depth_sync_viz')
        self.image_sub = Subscriber(self, Image, '/zed/zed_node/left_original/image_rect_color')
        self.camera_info_sub = Subscriber(self, CameraInfo, '/zed/zed_node/left_original/camera_info')
        self.depth_sub = Subscriber(self, Image, '/zed/zed_node/depth/depth_registered')
        self.bbox_pub = self.create_publisher(EmberBoundingBox3DArray, 'ember_detection/ember_bounding_boxes', 10)

        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.camera_info_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)
    
    # http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
    def callback(self, msg_image, msg_camera_info, msg_depth):
        # log timestamp
        self.get_logger().info(f"ImageDepthSyncVizSubscriber callback: {msg_image.header.stamp}")

        # create mock ember bounding box
        ember_bbox = EmberBoundingBox3D()
        ember_bbox.corner1 = Point(x=0.0, y=0.0, z=0.0)
        ember_bbox.corner2 = Point(x=1.0, y=0.0, z=0.0)
        ember_bbox.corner3 = Point(x=1.0, y=1.0, z=0.0)
        ember_bbox.corner4 = Point(x=0.0, y=1.0, z=0.0)
        ember_bbox.corner5 = Point(x=0.0, y=0.0, z=1.0)
        ember_bbox.corner6 = Point(x=1.0, y=0.0, z=1.0)
        ember_bbox.corner7 = Point(x=1.0, y=1.0, z=1.0)
        ember_bbox.corner8 = Point(x=0.0, y=1.0, z=1.0)
        
        ember_bbox_array = EmberBoundingBox3DArray()
        ember_bbox_array.header = Header()
        ember_bbox_array.header.stamp = msg_image.header.stamp
        ember_bbox_array.header.frame_id = "ember_bbox_frame"
        ember_bbox_array.boxes.append(ember_bbox)

        self.bbox_pub.publish(ember_bbox_array)


def main(args=None):
    rclpy.init(args=args)
    node = ImageDepthSyncVizSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
