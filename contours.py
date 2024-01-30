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
        return cv.contourArea(self.points)

    def circumference(self) -> float:
        return cv.arcLength(self.points, True)

    def center(self) -> Position:
        pass

    def centroid(self) -> Position:
        m = self.moments()
        return Position(m["m10"] / m["00"], m["01"] / m["00"])

    def moments(self) -> cv.typing.Moments:
        return cv.moments(self.points)

    def approximate_polygon(self, epsilon: float = None) -> "Contour":
        return Contour(cv.approxPolyDP(self.points, epsilon, True))

    def convex_hull(self) -> "Contour":
        return Contour(cv.convexHull(self.points))

    def is_convex(self) -> bool:
        return cv.isContourConvex(self.points)


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
