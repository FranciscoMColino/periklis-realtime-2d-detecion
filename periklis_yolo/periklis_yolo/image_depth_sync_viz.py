import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class ImageDepthSyncVizSubscriber(Node):
    def __init__(self):
        super().__init__('image_depth_sync_viz')
        self.image_sub = Subscriber(self, Image, '/zed/zed_node/left_original/image_rect_color')
        self.depth_sub = Subscriber(self, Image, '/zed/zed_node/depth/depth_registered')

        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)
    
    def callback(self, msg_image, msg_depth):
        bgr_image = CvBridge().imgmsg_to_cv2(msg_image, desired_encoding='bgr8')
        depth_image = CvBridge().imgmsg_to_cv2(msg_depth, desired_encoding='passthrough')
        
        resize_factor = 3

        # Resize the BGR image to half its original resolution
        bgr_image_half = cv2.resize(bgr_image, (bgr_image.shape[1] // resize_factor, bgr_image.shape[0] // resize_factor))
        
        # Resize the depth image to the same resolution as the resized BGR image
        depth_image_resized = cv2.resize(depth_image, (bgr_image_half.shape[1], bgr_image_half.shape[0]))

        max_range = 50.0
        depth_image_normalized = depth_image_resized / max_range * 255.0
        depth_image_normalized = np.uint8(depth_image_normalized)

        # Display the images in separate windows
        cv2.imshow('ZED RGB Image (Half Resolution)', bgr_image_half)
        cv2.imshow('ZED Depth Image (Normalized)', depth_image_normalized)
        cv2.waitKey(1)

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
