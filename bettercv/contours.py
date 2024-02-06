import cv2 as cv
from random import shuffle
from typing import Sequence
from dataclasses import dataclass
from numpy import ndarray, linalg, vstack

from .types import Position, Image
from .image import bgr, is_grayscale
from .colors import Color, max_sv, max_spaced_hues


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

    def is_close_to(self, other: "Contour", distance: float, jump: int = 10) -> bool:
        return any(linalg.norm(p1 - p2) < distance
                   for p1 in self.points[::jump]
                   for p2 in other.points[::jump])


def join_contours(contours: Sequence[Contour]) -> Contour:
    return Contour(vstack([contour.points for contour in contours])).convex_hull()


def find_contours(image: Image, external_only: bool = False) -> Sequence[Contour]:
    mode = cv.RETR_EXTERNAL if external_only else cv.RETR_TREE
    return tuple(map(Contour, cv.findContours(image, mode, cv.CHAIN_APPROX_SIMPLE)[0]))


def draw_contours(image: Image,
                  contours: Sequence[Contour],
                  color: Color = None,
                  thickness: int = 3) -> Image:
    canvas = bgr(image.copy()) if is_grayscale(image) else image.copy()
    if color:
        colors = [color] * len(contours)
    else:
        colors = list(map(max_sv, max_spaced_hues(len(contours))))
        shuffle(colors)
    for index, contour in enumerate(contours):
        cv.drawContours(canvas, [contour.points], 0, colors[index], thickness)
    return canvas
