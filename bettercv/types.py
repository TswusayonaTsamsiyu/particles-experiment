from numpy import ndarray
from collections import namedtuple

# This is just an alias for an `ndarray` (for now)
Image = ndarray


class Position(namedtuple("PositionBase", "x y")):
    """
    An integer position, in pixels, inside an image or a screen.
    This is a `namedtuple` (`pos[0] = pos.x`, `pos[1] = pos.y`).
    """

    def distance_to(self, other: "Position") -> float:
        """
        Computes the Euclidean distance to another position.
        """
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** .5

    def is_right_of(self, other: "Position") -> bool:
        """
        Checks whether the point is to the right of another point.
        """
        return self.x > other.x

    def is_higher_than(self, other: "Position") -> bool:
        """
        Checks if the point is higher than another point
        """
        return self.y < other.y


class Size(namedtuple("SizeBase", "width height")):
    """
    An integer size, in pixels, of an image or a screen.
    Supports scaling via multiplication and division, and addition of two sizes.
    This is also a `namedtuple` (`size[0] = size.width`, `size[1] = size.height`).
    """

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
        """
        The aspect ratio of the size (width / height).
        """
        return self.width / self.height

    @property
    def area(self) -> int:
        """
        The area of the rectangle defined by this size.
        """
        return self.width * self.height
