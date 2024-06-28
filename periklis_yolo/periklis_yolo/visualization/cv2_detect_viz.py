import cv2

class OpenCV2DetectVisualizer:
    def __init__(self, window_name='2D Detection'):
        self.window_name = window_name

    def draw_cv2_bounding_box(self, bbox, label, bgr_image):
        u1, v1, u2, v2 = bbox
        color = (0, 255, 0)
        cv2.rectangle(bgr_image, (u1, v1), (u2, v2), color, 2)
        cv2.putText(bgr_image, label, (u1, v1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)