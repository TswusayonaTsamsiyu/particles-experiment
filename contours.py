import cv2 as cv
from numpy import ndarray
from typing import Sequence
from dataclasses import dataclass

from utils import Position, Image, Color

GREEN = Color(0, 255, 0)


@dataclass
class Contour:
    points: ndarray

    def axes(self) -> Sequence[float]:
        return self.min_area_rect()[1]

    def width(self) -> float:
        return self.axes()[1]

    def length(self) -> float:
        return self.axes()[0]

    def area(self) -> float:
        return cv.contourArea(self.points)

    def circumference(self) -> float:
        return cv.arcLength(self.points, True)

    def center(self) -> Position:
        return Position(*self.min_area_rect()[0])

    def centroid(self) -> Position:
        m = self.moments()
        return Position(m["m10"] / m["m00"], m["m01"] / m["m00"])

    def moments(self) -> cv.typing.Moments:
        return cv.moments(self.points)

    def approximate_polygon(self, epsilon: float = None) -> "Contour":
        return Contour(cv.approxPolyDP(self.points, epsilon, True))

    def convex_hull(self) -> "Contour":
        return Contour(cv.convexHull(self.points))

    def is_convex(self) -> bool:
        return cv.isContourConvex(self.points)

    def min_area_rect(self) -> cv.typing.RotatedRect:
        return cv.minAreaRect(self.points)

    def fit_line(self, dist_type: int = cv.DIST_L2, reps: float = 0.01, aeps: float = 0.01):
        return cv.fitLine(self.points, dist_type, 0, reps, aeps)


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
