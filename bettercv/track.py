from datetime import timedelta
from dataclasses import dataclass
from typing import List, Iterator

from .video import Frame
from .contours import Contour


@dataclass
class Snapshot:
    frame: Frame
    contour: Contour

    def __repr__(self) -> str:
        return f"<Snapshot at {self.frame.index}, {self.frame.timestamp}>"

    def __str__(self) -> str:
        return repr(self).strip("<>")


class Track:
    def __init__(self):
        self.snapshots: List[Snapshot] = []

    def __iter__(self) -> Iterator[Snapshot]:
        return iter(self.snapshots)

    def __getitem__(self, index: int) -> Snapshot:
        return self.snapshots[index]

    @property
    def start(self) -> Snapshot:
        return self[0]

    @property
    def end(self) -> Snapshot:
        return self[-1]

    @property
    def extent(self) -> int:
        return self.end.frame.index - self.start.frame.index

    @property
    def duration(self) -> timedelta:
        return self.end.frame.timestamp - self.start.frame.timestamp

    def record(self, contour: Contour, frame: Frame) -> None:
        self.snapshots.append(Snapshot(frame, contour))
