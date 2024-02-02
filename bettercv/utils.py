from numpy import ndarray
from random import randint
from collections import namedtuple
from typing import Container, Callable

Image = ndarray
Position = namedtuple("Position", "x y")
SizeBase = namedtuple("SizeBase", "width height")
Color = namedtuple("Color", "red green blue")

MAX_PIXEL_VALUE = 255
MIN_PIXEL_VALUE = 0


class Size(SizeBase):
    def __mul__(self, factor: float) -> "Size":
        return Size(self.width * factor, self.height * factor)

    def __truediv__(self, factor: float) -> "Size":
        return Size(self.width / factor, self.height / factor)

    def __floordiv__(self, factor: float) -> "Size":
        return Size(self.width // factor, self.height // factor)

    def __add__(self, other: "Size") -> "Size":
        return Size(self.width + other.width, self.height + other.height)

    @property
    def aspect_ratio(self):
        return self.width / self.height


def random_intensity() -> int:
    return randint(MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)


def random_color() -> Color:
    return Color(random_intensity(), random_intensity(), random_intensity())


def exit_for(codes: Container[int]) -> Callable[[int], None]:
    return lambda key_code: exit() if key_code in codes else None
