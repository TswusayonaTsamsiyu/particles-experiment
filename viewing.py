import cv2
from typing import Tuple
from numpy import ndarray
from contextlib import contextmanager

ESC = 27
CLOSE_BTN = -1

SCALE_FACTOR = 0.8
SCREEN_SIZE = (1920, 1080)


def scale(dim: int) -> int:
    return int(dim * SCALE_FACTOR)


def resize_frame(frame: ndarray) -> ndarray:
    height, width = frame.shape
    factor = SCREEN_SIZE[1] / height * SCALE_FACTOR
    return cv2.resize(frame, None, fx=factor, fy=factor)


@contextmanager
def show_window(title: str, image: ndarray) -> None:
    height, width = image.shape
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
    adjust_window(title, (width, height))
    yield
    try:
        cv2.destroyWindow(title)
    except cv2.error:
        pass


def adjust_window(title: str,
                  size: Tuple[int, int],
                  position: Tuple[int, int] = (0, 0)) -> None:
    cv2.resizeWindow(title, *size)
    cv2.moveWindow(title, *position)


def display_frame(frame: ndarray, title: str = "Frame") -> None:
    with show_window(title, resize_frame(frame)):
        if cv2.waitKey(0) in [ESC, CLOSE_BTN]:
            exit()
