from dataclasses import dataclass

from bettercv.track import Track, Snapshot


@dataclass
class ParticleEvent:
    track: Track

    @property
    def start(self):
        return self.track.start

    @property
    def end(self):
        return self.track.end

    @property
    def extent(self):
        return self.track.extent

    @property
    def duration(self):
        return self.track.duration

    @property
    def best_snapshot(self) -> Snapshot:
        index = min(len(self.track.snapshots) - 1, 4)
        return self.track.snapshots[index]

    @property
    def width(self) -> float:
        return self.best_snapshot.contour.width

    @property
    def length(self) -> float:
        return self.best_snapshot.contour.length

    @property
    def angle(self):
        return self.best_snapshot.contour.angle

    @property
    def curvature(self):
        return NotImplemented

    @property
    def type(self) -> str:
        return NotImplemented
