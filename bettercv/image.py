import cv2 as cv
import numpy as np
from typing import Tuple

from .types import Image, Size
from .utils import MAX_PIXEL_VALUE, MIN_PIXEL_VALUE


def threshold_binary(image: Image, thresh: int) -> Image:
    return cv.threshold(image, thresh, MAX_PIXEL_VALUE, cv.THRESH_BINARY)[1]


def threshold_otsu(image: Image) -> Tuple[float, Image]:
    return cv.threshold(image, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE, cv.THRESH_BINARY + cv.THRESH_OTSU)


def threshold_adaptive(image: Image, method: int, block_size: int, cut: int) -> Image:
    return cv.adaptiveThreshold(image, MAX_PIXEL_VALUE, method, cv.THRESH_BINARY, block_size, cut)


def is_grayscale(image: Image) -> bool:
    return len(image.shape) == 2


def grayscale(image: Image) -> Image:
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def bgr(image: Image) -> Image:
    return cv.cvtColor(image, cv.COLOR_GRAY2BGR)


def blur(image: Image, ksize: Tuple[int, int]) -> Image:
    return cv.GaussianBlur(image, ksize, 0)


def denoise(image: Image) -> Image:
    # Slow...
    return cv.fastNlMeansDenoising(image)


def adjust_brightness_contrast(image: Image) -> Image:
    p5 = np.percentile(image, 5)
    p95 = np.percentile(image, 95)
    stretch = MAX_PIXEL_VALUE / (p95 - p5)
    return cv.convertScaleAbs(image, alpha=stretch, beta=-stretch * p5)


def subtract(image1: Image, image2: Image) -> Image:
    return cv.subtract(image1, image2)


def subtract_bg(image: Image, thresh: int) -> Image:
    # Doesn't work...
    return cv.createBackgroundSubtractorMOG2(varThreshold=thresh, detectShadows=False).apply(image)


def crop(image: Image, top: int, bottom: int) -> Image:
    return image[top:-bottom]


def scale(image: Image, factor: float) -> Image:
    return cv.resize(image, None, fx=factor, fy=factor)


def resize(image: Image, size: Size) -> Image:
    return cv.resize(image, size)
