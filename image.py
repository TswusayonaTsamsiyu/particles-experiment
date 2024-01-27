import cv2 as cv
import numpy as np
from numpy import ndarray
from typing import Tuple, Sequence

from utils import Color, Image

GREEN = Color(0, 255, 0)


def monochrome(image: Image) -> Image:
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def threshold(image: Image, thresh: int = None) -> Image:
    if thresh:
        return cv.threshold(image, thresh, 255, cv.THRESH_BINARY)[1]
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


def blur(image: Image, ksize: Tuple[int, int]) -> Image:
    return cv.GaussianBlur(image, ksize, 0)


def crop(image: Image, top: int, bottom: int) -> Image:
    return image[top:-bottom]


def adjust_brightness_contrast(image: Image) -> Image:
    p5 = np.percentile(image, 5)
    p95 = np.percentile(image, 95)
    stretch = 255 / (p95 - p5)
    return cv.convertScaleAbs(image, alpha=stretch, beta=-stretch * p5)


def denoise(image: Image) -> Image:
    # Slow...
    return cv.fastNlMeansDenoising(image)


def subtract_bg(image: Image, thresh: int) -> Image:
    # Doesn't work...
    return cv.createBackgroundSubtractorMOG2(varThreshold=thresh, detectShadows=False).apply(image)


def draw_contours(image: Image,
                  contours: Sequence[ndarray],
                  color: Color = GREEN,
                  thickness: int = 2) -> Image:
    rgb_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    return cv.drawContours(rgb_copy, contours, -1, color, thickness)
