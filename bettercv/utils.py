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
        return Size(int(self.width * factor), int(self.height * factor))

    def __truediv__(self, factor: float) -> "Size":
        return self // factor

    def __floordiv__(self, factor: float) -> "Size":
        return Size(self.width // factor, self.height // factor)

    def __add__(self, other: "Size") -> "Size":
        return Size(self.width + other.width, self.height + other.height)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


def random_intensity() -> int:
    return randint(MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)


def random_color() -> Color:
    return Color(random_intensity(), random_intensity(), random_intensity())


def exit_for(codes: Container[int]) -> Callable[[int], None]:
    return lambda key_code: exit() if key_code in codes else None


def distance(point1: Position, point2: Position) -> float:
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** .5
