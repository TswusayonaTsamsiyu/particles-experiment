from datetime import timedelta
from dataclasses import dataclass
from typing import List, Iterator, Union

from .video import Frame, Ref
from .contours import Contour


@dataclass
class Snapshot:
    ref: Ref
    contour: Contour

    def __repr__(self) -> str:
        return f"<Snapshot at {self.ref.index}, {self.ref.time} from {self.ref.video}>"

    def __str__(self) -> str:
        return repr(self).strip("<>")


class Track:
    def __init__(self):
        self.snapshots: List[Snapshot] = []

    def __iter__(self) -> Iterator[Snapshot]:
        return iter(self.snapshots)

    def __getitem__(self, index: Union[int, slice]) -> Snapshot:
        return self.snapshots[index]

    @property
    def start(self) -> Snapshot:
        return self[0]

    @property
    def end(self) -> Snapshot:
        return self[-1]

    @property
    def extent(self) -> int:
        return self.end.ref.index - self.start.ref.index

    @property
    def duration(self) -> timedelta:
        return self.end.ref.time - self.start.ref.time

    def record(self, contour: Contour, frame: Frame) -> None:
        self.snapshots.append(Snapshot(frame.ref, contour))
