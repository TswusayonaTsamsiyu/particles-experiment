import cv2 as cv
from typing import Union
from numpy import ndarray
from collections import namedtuple
from screeninfo import get_monitors
from contextlib import contextmanager

Image = ndarray
Position = namedtuple("Position", "x y")
Size = namedtuple("Size", "width height")

ESC = 27
CLOSE_BTN = -1

SCALE_FACTOR = 0.9

IMAGE = "image"
SCREEN = "screen"


def get_screen_size():
    primary_monitor = next(filter(lambda m: m.is_primary, get_monitors()))
    return Size(primary_monitor.width, primary_monitor.height)


def adjust_size_to_screen(image: Image) -> Size:
    height, width = image.shape
    aspect_ratio = width / height
    screen_size = get_screen_size()
    screen_aspect_ratio = screen_size.width / screen_size.height
    if aspect_ratio > screen_aspect_ratio:
        new_width = screen_size.width * SCALE_FACTOR
        new_height = height * new_width / width
    else:
        new_height = screen_size.height * SCALE_FACTOR
        new_width = width * new_height / height
    return Size(int(new_width), int(new_height))


def show_window(image: Image,
                title: str = "Image",
                size: Union[Size, SCREEN, IMAGE] = SCREEN,
                position: Position = None) -> None:
    if size == IMAGE:
        cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
        cv.imshow(title, image)
    else:
        cv.namedWindow(title, cv.WINDOW_NORMAL)
        if size == SCREEN:
            size = adjust_size_to_screen(image)
        cv.imshow(title, cv.resize(image, tuple(size)))
        cv.resizeWindow(title, *size)
    if position:
        cv.moveWindow(title, *position)


@contextmanager
def window_control():
    yield
    key_code = cv.waitKey(0)
    cv.destroyAllWindows()
    if key_code in [ESC, CLOSE_BTN]:
        exit()
