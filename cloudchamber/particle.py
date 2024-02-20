from typing import Tuple
from dataclasses import dataclass

from bettercv.track import Track, Snapshot


@dataclass
class Particle:
    ref: Tuple[int, int]
    snapshot: Snapshot

    @property
    def start(self):
        return self.ref[0]

    @property
    def end(self):
        return self.ref[1]

    @property
    def length(self):
        return self.snapshot.contour.length

    @property
    def width(self):
        return self.snapshot.contour.width

    @property
    def angle(self):
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
        return cls((track.start.frame.index, track.end.frame.index), best_snapshot)
