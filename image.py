import cv2 as cv
import numpy as np
from typing import Tuple

from utils import Image


def monochrome(image: Image) -> Image:
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def threshold(image: Image, thresh: int = None) -> Image:
    return cv.threshold(image, thresh, 255, cv.THRESH_BINARY)[1]


def otsu_threshold(image: Image) -> Image:
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


def adaptive_threshold(image: Image, adaptive_method: int = cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                       threshold_type: int = cv.THRESH_BINARY, block_size: int = 11, cut: int = 0) -> Image:
    return cv.adaptiveThreshold(image, 255, adaptive_method, threshold_type, block_size, cut)


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


def scale(image: Image, factor: float) -> Image:
    return cv.resize(image, None, fx=factor, fy=factor)
