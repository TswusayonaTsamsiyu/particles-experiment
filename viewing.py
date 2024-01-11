import cv2
from numpy import ndarray

TITLE = "Frame"
RESIZE_FACTOR = 0.4


def display_frame(frame: ndarray):
    frame = cv2.resize(frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    cv2.imshow(TITLE, frame)
    cv2.waitKey(0)
    cv2.destroyWindow(TITLE)
