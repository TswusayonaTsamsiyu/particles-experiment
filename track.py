from datetime import timedelta
from typing import MutableSequence, Tuple

from bettercv.video import Frame
from bettercv.contours import Contour


class Track:
    def __init__(self, contours: MutableSequence[Contour], start: Frame):
        self.contours = contours
        self.start = start
        self.end = start

    @property
    def duration(self) -> Tuple[float, timedelta]:
        return (self.end.index - self.start.index,
                self.end.timestamp - self.start.timestamp)

    @property
    def _relevant_contour_index(self) -> int:
        return min(len(self.contours) - 1, 4)

    @property
    def relevant_contour(self) -> Contour:
        return self.contours[self._relevant_contour_index]

    @property
    def relevant_frame_index(self) -> int:
        return self.start.index + self._relevant_contour_index

    @property
    def type(self) -> str:
        return "Don't know yet"

    def append(self, contour: Contour) -> None:
        self.contours.append(contour)
