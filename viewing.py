import cv2
from numpy import ndarray

ESC = 27


def display_frame(frame: ndarray, title: str = "Frame") -> None:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.moveWindow(title, 0, 0)
    cv2.imshow(title, frame)
    key_code = cv2.waitKey(0)
    cv2.destroyWindow(title)
    if key_code == ESC:
        exit()
