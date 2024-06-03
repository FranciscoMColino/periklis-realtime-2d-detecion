import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

class YoloV8Subscriber(Node):
    def __init__(self):
        super().__init__('yolov8_subscriber')
        self.subscription = self.create_subscription(Image, '/zed/zed_node/left_original/image_rect_color', self.listener_callback, 10)
        self.cv_bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')  # Load the YOLOv8 model

    def listener_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.resize(cv_image, (640, 480))
        
        # Run YOLOv8 inference
        results = self.model(cv_image)[0]
        
        # Draw bounding boxes and labels on the image
        for result in results.boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            confidence = result.conf[0].item()
            class_id = int(result.cls[0].item())
            label = f'{self.model.names[class_id]}: {confidence:.2f}'
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow('YOLOv8 Detection', cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloV8Subscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
