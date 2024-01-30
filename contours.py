import cv2 as cv
from numpy import ndarray
from typing import Sequence
from dataclasses import dataclass

from utils import Position, Image, Color

GREEN = Color(0, 255, 0)


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


def find_contours(image: Image) -> Sequence[Contour]:
    return tuple(map(Contour, cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]))


def draw_contours(image: Image,
                  contours: Sequence[Contour],
                  color: Color = GREEN,
                  thickness: int = 2) -> Image:
    rgb_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    return cv.drawContours(rgb_copy, tuple(contour.points for contour in contours), -1, color, thickness)
