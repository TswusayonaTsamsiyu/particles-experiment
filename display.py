import cv2 as cv
from typing import Callable
from dataclasses import dataclass
from screeninfo import get_monitors
from contextlib import contextmanager

from utils import Image, Size, Position

SCALE_FACTOR = 0.9
WINDOW_SEP = 5


@dataclass
class Window:
    title: str
    size: Size
    position: Position

    def destroy(self):
        cv.destroyWindow(self.title)


def get_screen_size():
    primary_monitor = next(filter(lambda m: m.is_primary, get_monitors()))
    return Size(primary_monitor.width, primary_monitor.height)


def get_image_size(image: Image) -> Size:
    height, width, *channels = image.shape
    return Size(width, height)


def get_aspect_ratio(size: Size) -> float:
    return size.width / size.height


def fit_to_screen(image: Image) -> Image:
    image_size = get_image_size(image)
    screen_size = get_screen_size()
    if get_aspect_ratio(image_size) > get_aspect_ratio(screen_size):
        new_width = screen_size.width * SCALE_FACTOR
        new_height = image_size.height * new_width / image_size.width
    else:
        new_height = screen_size.height * SCALE_FACTOR
        new_width = image_size.width * new_height / image_size.height
    return cv.resize(image, (int(new_width), int(new_height)))


def right_of(window: Window) -> Position:
    return Position(window.position.x + window.size.width + WINDOW_SEP, window.position.y)


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
