import cv2
from numpy import ndarray

ESC = 27

SCALE_FACTOR = 0.8
SCREEN_SIZE = (1920, 1080)


def scale(dim: int) -> int:
    return int(dim * SCALE_FACTOR)


def resize_frame(frame: ndarray) -> ndarray:
    height, width = frame.shape
    factor = SCREEN_SIZE[1] / height * SCALE_FACTOR
    return cv2.resize(frame, None, fx=factor, fy=factor)


def create_window(frame: ndarray, title: str) -> None:
    height, width = frame.shape
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, frame)
    cv2.resizeWindow(title, width, height)
    cv2.moveWindow(title, 0, 0)


def display_frame(frame: ndarray, title: str = "Frame") -> None:
    create_window(resize_frame(frame), title)
    key_code = cv2.waitKey(0)
    cv2.destroyWindow(title)
    if key_code == ESC:
        exit()
