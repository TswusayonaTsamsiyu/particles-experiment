from datetime import timedelta
from dataclasses import dataclass
from typing import List, Iterator

from .video import Frame
from .contours import Contour


@dataclass
class Snapshot:
    index: int
    timestamp: timedelta
    contour: Contour

    def __repr__(self) -> str:
        return f"<Snapshot at {self.index}, {self.timestamp}>"

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
        return self.snapshots[0]

    @property
    def end(self) -> Snapshot:
        return self.snapshots[-1]

    @property
    def extent(self) -> int:
        return self.end.index - self.start.index

    @property
    def duration(self) -> timedelta:
        return self.end.timestamp - self.start.timestamp

    def record(self, contour: Contour, frame: Frame) -> None:
        self.snapshots.append(Snapshot(frame.index, frame.timestamp, contour))