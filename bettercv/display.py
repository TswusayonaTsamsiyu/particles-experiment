import cv2 as cv
from typing import Iterable
from functools import cache
from screeninfo import get_monitors

from .types import Image, Size, Position
from .image import resize, get_image_size

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
        self._shown = False

    def fit_to_screen(self) -> "Window":
        screen_size = get_screen_size()
        if self.size.aspect_ratio > screen_size.aspect_ratio:
            scaling = screen_size.width / self.size.width
        else:
            scaling = screen_size.height / self.size.height
        return Window(self.image,
                      self.title,
                      self.size * scaling * SCALE_FACTOR,
                      self.position)

    def show(self, destroy: bool = True, timeout: int = 0) -> int:
        self._show()
        key_code = cv.waitKey(timeout)
        if destroy:
            destroy_window(self)
        return key_code

    def destroy(self) -> None:
        if self._shown:
            cv.destroyWindow(self.title)
        self._shown = False

    def _show(self) -> None:
        if not self._shown:
            cv.namedWindow(self.title, cv.WINDOW_NORMAL)
            cv.imshow(self.title, resize(self.image, self.size))
            cv.resizeWindow(self.title, *self.size)
            cv.moveWindow(self.title, *self._fix_position())
        cv.setWindowProperty(self.title, cv.WND_PROP_VISIBLE, 1)
        self._shown = True

    def _fix_position(self) -> Position:
        return Position(max(0, self.position.x - self.size.width // 2),
                        max(0, self.position.y - self.size.height // 2))


def show(windows: Iterable[Window], destroy: bool = True, timeout: int = 0) -> int:
    windows = tuple(windows)
    for window in windows:
        show_window(window)
    key_code = cv.waitKey(timeout)
    if destroy:
        for window in windows:
            destroy_window(window)
    return key_code


def show_window(window: Window) -> None:
    cv.namedWindow(window.title, cv.WINDOW_NORMAL)
    cv.imshow(window.title, resize(window.image, window.size))
    cv.resizeWindow(window.title, *window.size)
    cv.moveWindow(window.title, *fix_position(window))
    cv.setWindowProperty(window.title, cv.WND_PROP_VISIBLE, 1)


def fix_position(window: Window) -> Position:
    return Position(max(0, window.position.x - window.size.width // 2),
                    max(0, window.position.y - window.size.height // 2))


def destroy_window(window: Window) -> None:
    try:
        cv.destroyWindow(window.title)
    except cv.error:
        pass


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
