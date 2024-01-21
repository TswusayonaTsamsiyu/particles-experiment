import cv2
import numpy as np
from typing import Tuple
from numpy import ndarray


def monochrome(frame: ndarray) -> ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def threshold(frame: ndarray, thresh: int = None) -> ndarray:
    if thresh:
        return cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)[1]
    return cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]


def blur(frame: ndarray, ksize: Tuple[int, int]) -> ndarray:
    return cv2.GaussianBlur(frame, ksize, 0)


def crop(frame: ndarray, top: int, bottom: int) -> ndarray:
    return frame[top:-bottom]


def adjust_brightness_contrast(frame: ndarray) -> ndarray:
    p5 = np.percentile(frame, 5)
    p95 = np.percentile(frame, 95)
    stretch = 255 / (p95 - p5)
    return cv2.convertScaleAbs(frame, alpha=stretch, beta=-stretch * p5)


def denoise(frame: ndarray) -> ndarray:
    # Slow...
    return cv2.fastNlMeansDenoising(frame)


def subtract_bg(frame: ndarray, thresh: int) -> ndarray:
    # Doesn't work...
    return cv2.createBackgroundSubtractorMOG2(varThreshold=thresh, detectShadows=False).apply(frame)
