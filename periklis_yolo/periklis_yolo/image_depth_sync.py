import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer

class ImageDepthSyncSubscriber(Node):
    def __init__(self):
        super().__init__('sync_subscriber')
        self.image_sub = Subscriber(self, Image, '/zed/zed_node/left/image_rect_color')
        self.depth_sub = Subscriber(self, Image, '/zed/zed_node/depth/depth_registered')

        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)
    
    def callback(self, msg_image, msg_depth):
        # print the timestamp of the image and depth messages
        self.get_logger().info(f'Depth timestamp: {msg_image.header.stamp}')
        self.get_logger().info(f'Depth timestamp: {msg_depth.header.stamp}')
        self.get_logger().info('')


def main(args=None):
    rclpy.init(args=args)
    node = ImageDepthSyncSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
