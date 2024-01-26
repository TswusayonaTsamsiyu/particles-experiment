import cv2 as cv
import numpy as np
from numpy import ndarray
from typing import Tuple, Sequence

from utils import Color

GREEN = Color(0, 255, 0)


def monochrome(frame: ndarray) -> ndarray:
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


def threshold(frame: ndarray, thresh: int = None) -> ndarray:
    if thresh:
        return cv.threshold(frame, thresh, 255, cv.THRESH_BINARY)[1]
    return cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


def blur(frame: ndarray, ksize: Tuple[int, int]) -> ndarray:
    return cv.GaussianBlur(frame, ksize, 0)


def crop(frame: ndarray, top: int, bottom: int) -> ndarray:
    return frame[top:-bottom]


def adjust_brightness_contrast(frame: ndarray) -> ndarray:
    p5 = np.percentile(frame, 5)
    p95 = np.percentile(frame, 95)
    stretch = 255 / (p95 - p5)
    return cv.convertScaleAbs(frame, alpha=stretch, beta=-stretch * p5)


def denoise(frame: ndarray) -> ndarray:
    # Slow...
    return cv.fastNlMeansDenoising(frame)


def subtract_bg(frame: ndarray, thresh: int) -> ndarray:
    # Doesn't work...
    return cv.createBackgroundSubtractorMOG2(varThreshold=thresh, detectShadows=False).apply(frame)


def draw_contours(frame: ndarray,
                  contours: Sequence[ndarray],
                  color: Color = GREEN,
                  thickness: int = 2) -> ndarray:
    rgb_copy = cv.cvtColor(frame.copy(), cv.COLOR_GRAY2BGR)
    return cv.drawContours(rgb_copy, contours, -1, color, thickness)
