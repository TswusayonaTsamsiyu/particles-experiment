from typing import Tuple
from dataclasses import dataclass

from bettercv.video import Ref
from bettercv.track import Track, Snapshot


@dataclass
class Particle:
    range: Tuple[Ref, Ref]
    snapshot: Snapshot

    @property
    def start(self) -> int:
        return self.range[0].index

    @property
    def end(self) -> int:
        return self.range[1].index

    @property
    def length(self) -> float:
        return self.snapshot.contour.length

    @property
    def width(self) -> float:
        return self.snapshot.contour.width

    @property
    def angle(self) -> float:
        return self.snapshot.contour.angle

    @property
    def curvature(self):
        return NotImplemented

    @property
    def intensity(self):
        return NotImplemented

    @property
    def type(self):
        return NotImplemented

    @classmethod
    def from_track(cls, track: Track) -> "Particle":
        best_snapshot = track.snapshots[min(len(track.snapshots) - 1, 4)]
        return cls((track.start.ref, track.end.ref), best_snapshot)
