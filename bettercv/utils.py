from numpy import ndarray
from random import randint
from collections import namedtuple
from typing import Container, Callable

Image = ndarray
Position = namedtuple("Position", "x y")
Size = namedtuple("Size", "width height")
Color = namedtuple("Color", "red green blue")

MAX_PIXEL_VALUE = 255
MIN_PIXEL_VALUE = 0


def random_intensity() -> int:
    return randint(MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)


def random_color() -> Color:
    return Color(random_intensity(), random_intensity(), random_intensity())


def exit_for(codes: Container[int]) -> Callable[[int], None]:
    return lambda key_code: exit() if key_code in codes else None
