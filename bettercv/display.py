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
        self._size = size
        self._center = position

    def show(self, auto_close: bool = True, timeout: int = 0) -> int:
        _show_window(self)
        key_code = cv.waitKey(timeout)
        if auto_close:
            self.close()
        return key_code

    def close(self) -> None:
        try:
            cv.destroyWindow(self.title)
        except cv.error:
            pass

    def fit_to_screen(self) -> "Window":
        screen_size = get_screen_size()
        if self.size.aspect_ratio > screen_size.aspect_ratio:
            scaling = screen_size.width / self.size.width
        else:
            scaling = screen_size.height / self.size.height
        return Window(self.image,
                      self.title,
                      self.size * scaling * SCALE_FACTOR,
                      self.center)

    def bring_to_front(self) -> "Window":
        cv.setWindowProperty(self.title, cv.WND_PROP_TOPMOST, 1)
        return self

    @property
    def size(self) -> Size:
        return self._size or get_image_size(self.image)

    @property
    def center(self) -> Position:
        return self._center or screen_center()

    @property
    def position(self) -> Position:
        return Position(max(0, self.center.x - self.size.width // 2),
                        max(0, self.center.y - self.size.height // 2))


def _show_window(window: Window) -> Window:
    cv.namedWindow(window.title, cv.WINDOW_NORMAL)
    cv.imshow(window.title, resize(window.image, window.size))
    cv.resizeWindow(window.title, *window.size)
    cv.moveWindow(window.title, *window.position)
    return window.bring_to_front()


def show(windows: Iterable[Window], auto_close: bool = True, timeout: int = 0) -> int:
    windows = tuple(windows)
    for window in windows:
        _show_window(window)
    key_code = cv.waitKey(timeout)
    if auto_close:
        close(windows)
    return key_code


def close(windows: Iterable[Window]) -> None:
    for window in windows:
        window.close()


def close_all() -> None:
    cv.destroyAllWindows()


def right_of(window: Window) -> Position:
    return Position(window.center.x + window.size.width // 2 + WINDOW_SEP, window.center.y)


def left_of(window: Window) -> Position:
    return Position(window.center.x - window.size.width // 2 - WINDOW_SEP, window.center.y)


@cache
def get_screen_size() -> Size:
    primary_monitor = next(filter(lambda m: m.is_primary, get_monitors()))
    return Size(primary_monitor.width, primary_monitor.height)


@cache
def screen_center() -> Position:
    return Position(*(get_screen_size() // 2))
