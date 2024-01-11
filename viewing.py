import cv2
from numpy import ndarray

TITLE = "Frame"
SCREEN_HEIGHT = 1080


def resize(frame: ndarray) -> ndarray:
    factor = SCREEN_HEIGHT / frame.shape[0] * 0.8
    return cv2.resize(frame, None, fx=factor, fy=factor)


def display_frame(frame: ndarray, title: str = TITLE) -> None:
    cv2.imshow(title, resize(frame))
    cv2.waitKey(0)
    cv2.destroyWindow(title)
