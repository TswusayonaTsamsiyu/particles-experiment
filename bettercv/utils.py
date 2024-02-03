from numpy import ndarray
from collections import namedtuple
from typing import Container, Callable

Image = ndarray
Position = namedtuple("Position", "x y")

MAX_PIXEL_VALUE = 255
MIN_PIXEL_VALUE = 0


class Size(namedtuple("SizeBase", "width height")):
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


def exit_for(codes: Container[int]) -> Callable[[int], None]:
    return lambda key_code: exit() if key_code in codes else None


def distance(point1: Position, point2: Position) -> float:
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** .5
