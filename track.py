from typing import MutableSequence

from video import Frame
from contours import Contour


class Track:
    def __init__(self, contours: MutableSequence[Contour], start: Frame):
        self.contours = contours
        self.start = start
        self.end = start

    def relevant_contour(self) -> Contour:
        index = min(len(self.contours) - 1, 4)
        return self.contours[index]

    def type(self) -> str:
        pass

    def append(self, contour: Contour) -> None:
        self.contours.append(contour)
