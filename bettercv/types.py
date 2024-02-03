from numpy import ndarray
from collections import namedtuple

Image = ndarray
Position = namedtuple("Position", "x y")


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
