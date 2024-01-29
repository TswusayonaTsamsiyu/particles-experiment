from numpy import ndarray
from dataclasses import dataclass

from utils import Position


@dataclass
class Contour:
    points: ndarray

    def axes(self):
        pass

    def breadth(self) -> float:
        pass

    def length(self) -> float:
        pass

    def area(self) -> float:
        pass

    def center(self) -> Position:
        pass


def contour_distance(contour1: Contour, contour2: Contour) -> float:
    pass
