from typing import Tuple
from dataclasses import dataclass

from bettercv.image import mean
from bettercv.video import Ref, Video
from bettercv.track import Track, Snapshot

from .config import Config
from .processing import preprocess


@dataclass
class Particle:
    range: Tuple[Ref, Ref]
    snapshot: Snapshot

    @property
    def start(self) -> Ref:
        return self.range[0]

    @property
    def end(self) -> Ref:
        return self.range[1]

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
        with Video(self.snapshot.ref.video) as video:
            frame = preprocess(video[self.snapshot.ref.index], Config)
        return mean(frame.image, self.snapshot.contour.create_mask(frame.image.shape))[0]

    @property
    def type(self):
        return NotImplemented

    @classmethod
    def from_track(cls, track: Track) -> "Particle":
        best_snapshot = track.snapshots[min(len(track.snapshots) - 1, 4)]
        return cls((track.start.ref, track.end.ref), best_snapshot)
