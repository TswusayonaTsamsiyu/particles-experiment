from numpy import ndarray
from collections import namedtuple
from typing import Container, Callable

Image = ndarray
Contour = ndarray
Position = namedtuple("Position", "x y")
Size = namedtuple("Size", "width height")
Color = namedtuple("Color", "red green blue")


def exit_for(codes: Container[int]) -> Callable[[int], None]:
    return lambda key_code: exit() if key_code in codes else None
