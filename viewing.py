import cv2
import pyautogui
from numpy import ndarray
from typing import Tuple, Union
from contextlib import contextmanager

Image = ndarray
Size = pyautogui.Size
Position = pyautogui.Point

ESC = 27
CLOSE_BTN = -1

SCALE_FACTOR = 0.9
SCREEN_SIZE = pyautogui.size()
SCREEN_ASPECT_RATIO = SCREEN_SIZE.width / SCREEN_SIZE.height

IMAGE = "image"
SCREEN = "screen"


def adjust_size_to_screen(image: Image) -> Size:
    height, width = image.shape
    aspect_ratio = width / height
    if aspect_ratio > SCREEN_ASPECT_RATIO:
        new_width = SCREEN_SIZE.width * SCALE_FACTOR
        new_height = height * new_width / width
    else:
        new_height = SCREEN_SIZE.height * SCALE_FACTOR
        new_width = width * new_height / height
    return Size(int(new_width), int(new_height))


def show_window(image: Image,
                title: str = "Image",
                size: Union[Size, SCREEN, IMAGE] = SCREEN,
                position: Position = None) -> None:
    if size == IMAGE:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(title, image)
    else:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        if size == SCREEN:
            size = adjust_size_to_screen(image)
        cv2.imshow(title, cv2.resize(image, tuple(size)))
        cv2.resizeWindow(title, *size)
    if position:
        cv2.moveWindow(title, *position)


@contextmanager
def window_control():
    yield
    key_code = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key_code in [ESC, CLOSE_BTN]:
        exit()
