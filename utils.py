from numpy import ndarray
from collections import namedtuple

Image = ndarray
Contour = ndarray
Position = namedtuple("Position", "x y")
Size = namedtuple("Size", "width height")
Color = namedtuple("Color", "red green blue")
