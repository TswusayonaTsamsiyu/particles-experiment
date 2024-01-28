import cv2 as cv
from typing import Callable
from dataclasses import dataclass
from screeninfo import get_monitors
from contextlib import contextmanager

from image import scale
from utils import Image, Size, Position

SCALE_FACTOR = 0.9
WINDOW_SEP = 5


@dataclass
class Window:
    title: str
    size: Size
    position: Position


def get_screen_size():
    primary_monitor = next(filter(lambda m: m.is_primary, get_monitors()))
    return Size(primary_monitor.width, primary_monitor.height)


def get_image_size(image: Image) -> Size:
    height, width, *channels = image.shape
    return Size(width, height)


def get_aspect_ratio(size: Size) -> float:
    return size.width / size.height


def fit_to_screen(image: Image) -> Image:
    screen_size = get_screen_size()
    image_size = get_image_size(image)
    if get_aspect_ratio(image_size) > get_aspect_ratio(screen_size):
        scaling = screen_size.width / image_size.width
    else:
        scaling = screen_size.height / image_size.height
    return scale(image, scaling * SCALE_FACTOR)


def right_of(window: Window) -> Position:
    return Position(window.position.x + window.size.width + WINDOW_SEP, window.position.y)


def screen_center() -> Position:
    return Position(*map(lambda x: x // 2, get_screen_size()))


def show_window(image: Image,
                title: str = "Image",
                position: Position = None) -> Window:
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.imshow(title, image)
    size = get_image_size(image)
    cv.resizeWindow(title, *size)
    if position:
        cv.moveWindow(title, *position)
    return Window(title, size, position)


@contextmanager
def window_control(handle_key: Callable[[int], None] = None) -> None:
    yield
    key_code = cv.waitKey(0)
    cv.destroyAllWindows()
    if handle_key:
        handle_key(key_code)


def show_single_window(image: Image,
                       title: str = "Image",
                       position: Position = None) -> None:
    with window_control():
        show_window(image, title, position)
