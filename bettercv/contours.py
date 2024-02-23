import cv2 as cv
from random import shuffle
from typing import Sequence
from numpy import ndarray, vstack
from dataclasses import dataclass
from functools import cached_property

from .types import Position, Image
from .image import bgr, is_grayscale
from .colors import Color, max_sv, max_spaced_hues


@dataclass
class Contour:
    points: ndarray

    @property
    def axes(self) -> Sequence[float]:
        return self.min_area_rect[1]

    @property
    def width(self) -> float:
        return min(self.axes)

    @property
    def length(self) -> float:
        return max(self.axes)

    @property
    def angle(self):
        return self.min_area_rect[2] + (90 if self.axes[0] < self.axes[1] else 0)

    @cached_property
    def area(self) -> float:
        return cv.contourArea(self.points)

    @cached_property
    def circumference(self) -> float:
        return cv.arcLength(self.points, True)

    @property
    def center(self) -> Position:
        return Position(*self.min_area_rect[0])

    @property
    def centroid(self) -> Position:
        return Position(self.moments["m10"] / self.moments["m00"],
                        self.moments["m01"] / self.moments["m00"])

    @cached_property
    def moments(self) -> cv.typing.Moments:
        return cv.moments(self.points)

    def approximate_polygon(self, epsilon: float = None) -> "Contour":
        return Contour(cv.approxPolyDP(self.points, epsilon, True))

    def convex_hull(self) -> "Contour":
        return Contour(cv.convexHull(self.points))

    @cached_property
    def is_convex(self) -> bool:
        return cv.isContourConvex(self.points)

    @cached_property
    def min_area_rect(self) -> cv.typing.RotatedRect:
        return cv.minAreaRect(self.points)

    def fit_line(self, dist_type: int = cv.DIST_L2, reps: float = 0.01, aeps: float = 0.01):
        return cv.fitLine(self.points, dist_type, 0, reps, aeps)

    def is_close_to(self, other: "Contour", distance: float, jump: int = 10) -> bool:
        return any(cv.norm(p1 - p2) < distance
                   for p1 in self.points[::jump]
                   for p2 in other.points[::jump])


def join_contours(contours: Sequence[Contour]) -> Contour:
    return Contour(vstack([contour.points for contour in contours])).convex_hull()


def join_close_contours(contours: Sequence[Contour], closeness: int) -> Sequence[Contour]:
    groups = []
    for i, c1 in enumerate(contours):
        close_indices = [j for j, c2 in tuple(enumerate(contours))[i + 1:]
                         if c2.is_close_to(c1, closeness)]
        new_group = set(close_indices + [i])
        disjoint_groups = []
        for group in groups:
            if new_group.intersection(group):
                new_group = new_group.union(group)
            else:
                disjoint_groups.append(group)
        groups = disjoint_groups + [new_group]
    return [join_contours([contours[index] for index in group]) for group in groups]


def find_contours(image: Image, external_only: bool = False) -> Sequence[Contour]:
    mode = cv.RETR_EXTERNAL if external_only else cv.RETR_TREE
    return tuple(map(Contour, cv.findContours(image, mode, cv.CHAIN_APPROX_SIMPLE)[0]))


def draw_contours(image: Image,
                  contours: Sequence[Contour],
                  color: Color = None,
                  thickness: int = 3,
                  fill: bool = False) -> Image:
    canvas = bgr(image.copy()) if is_grayscale(image) else image.copy()
    if color:
        colors = [color] * len(contours)
    else:
        colors = list(map(max_sv, max_spaced_hues(len(contours))))
        shuffle(colors)
    for index, contour in enumerate(contours):
        cv.drawContours(canvas, [contour.points], 0, colors[index], thickness if not fill else -1)
    return canvas
