from typing import MutableSequence

from bettercv.video import Frame
from bettercv.contours import Contour


class Track:
    def __init__(self, contours: MutableSequence[Contour], start: Frame):
        self.contours = contours
        self.start = start
        self.end = start

    def __len__(self):
        return len(self.contours)

    def length(self) -> int:
        return len(self.contours)

    def _relevant_contour_index(self) -> int:
        return min(len(self.contours) - 1, 4)

    def relevant_contour(self) -> Contour:
        return self.contours[self._relevant_contour_index()]

    def relevant_frame_index(self) -> int:
        return self.start.index + self._relevant_contour_index()

    def type(self) -> str:
        pass

    def append(self, contour: Contour) -> None:
        self.contours.append(contour)

