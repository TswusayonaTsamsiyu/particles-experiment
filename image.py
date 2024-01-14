import cv2
from typing import Tuple
from numpy import ndarray


def monochrome(frame: ndarray) -> ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def threshold(frame: ndarray, thresh: int) -> ndarray:
    return 255 - cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)[1]


def blur(frame: ndarray, ksize: Tuple[int, int]) -> ndarray:
    return cv2.GaussianBlur(frame, ksize, 0)


def crop(frame: ndarray, top: int, bottom: int) -> ndarray:
    return frame[top:-bottom]


def subtract_bg(frame: ndarray, thresh: int) -> ndarray:
    return cv2.createBackgroundSubtractorMOG2(varThreshold=thresh, detectShadows=False).apply(frame)
