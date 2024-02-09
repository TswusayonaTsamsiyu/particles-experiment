import cv2 as cv
from typing import Iterable
from functools import cache
from screeninfo import get_monitors

from .image import resize
from .types import Image, Size, Position

SCALE_FACTOR = 0.85
WINDOW_SEP = 5


class Window:
    def __init__(self,
                 image: Image,
                 title: str,
                 size: Size = None,
                 position: Position = None) -> None:
        self.image = image
        self.title = title
        self.size = size or get_image_size(image)
        self.position = position or screen_center()


def show(windows: Iterable[Window], destroy=True) -> int:
    windows = tuple(windows)
    for window in windows:
        show_window(window)
    key_code = wait_key()
    if destroy:
        for window in windows:
            destroy_window(window)
    return key_code


def show_window(window: Window) -> None:
    cv.namedWindow(window.title, cv.WINDOW_NORMAL)
    cv.imshow(window.title, resize(window.image, window.size))
    cv.resizeWindow(window.title, *window.size)
    cv.moveWindow(window.title, *fix_position(window))


def fix_position(window: Window) -> Position:
    return Position(max(0, window.position.x - window.size.width // 2),
                    max(0, window.position.y - window.size.height // 2))


def destroy_window(window: Window) -> None:
    try:
        cv.destroyWindow(window.title)
    except cv.error:
        pass


def wait_key() -> int:
    return cv.waitKey(0)


def get_image_size(image: Image) -> Size:
    height, width, *channels = image.shape
    return Size(width, height)


def fit_to_screen(window: Window) -> Window:
    screen_size = get_screen_size()
    if window.size.aspect_ratio > window.size.aspect_ratio:
        scaling = screen_size.width / window.size.width
    else:
        scaling = screen_size.height / window.size.height
    return Window(window.image,
                  window.title,
                  window.size * scaling * SCALE_FACTOR,
                  window.position)


def right_of(window: Window) -> Position:
    return Position(window.position.x + window.size.width // 2 + WINDOW_SEP, window.position.y)


def left_of(window: Window) -> Position:
    return Position(window.position.x - window.size.width // 2 - WINDOW_SEP, window.position.y)


@cache
def get_screen_size() -> Size:
    primary_monitor = next(filter(lambda m: m.is_primary, get_monitors()))
    return Size(primary_monitor.width, primary_monitor.height)


@cache
def screen_center() -> Position:
    return Position(*(get_screen_size() // 2))
