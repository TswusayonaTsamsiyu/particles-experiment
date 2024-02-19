import pandas as pd
from pathlib import Path
from typing import List, Tuple, Iterable

from bettercv.track import Snapshot
from bettercv.contours import Contour

from cloudchamber.particle import ParticleTrack

from root import ROOT_PATH

BG_RADIATION_PATH = ROOT_PATH / "Background"
ROD_RADIATION_PATH = ROOT_PATH / "Rod"

CSV_PATH = ROOT_PATH / "csv"

_COLUMNS = ("Width", "Length", "Angle",
            "Start Index", "End Index",
            "Snapshot Index", "Contour")


def _is_video(path: Path) -> bool:
    return path.suffix.lower() == ".mp4"


def _get_videos(path: Path) -> List[Path]:
    return sorted(filter(_is_video, path.iterdir()))


def get_bg_videos() -> List[Path]:
    return _get_videos(BG_RADIATION_PATH)


def get_rod_videos() -> List[Path]:
    return _get_videos(ROD_RADIATION_PATH)


def _csv_contour(contour: Contour) -> str:
    return ":".join(",".join(map(str, point[0])) for point in contour.points)


def _csv_row(particle: ParticleTrack) -> Tuple:
    return (particle.width, particle.length, particle.angle,
            particle.start, particle.end,
            particle.snapshot.frame.index, _csv_contour(particle.snapshot.contour))


def save_particles(particles: Iterable[ParticleTrack], path: Path) -> None:
    pd.DataFrame(map(_csv_row, particles), columns=_COLUMNS).to_csv(path, index=False)


def read_particles(path: Path) -> List[ParticleTrack]:
    return [ParticleTrack((row[3], row[4]), Snapshot(row[5], Contour(row[6])))
            for row in pd.read_csv(path).iterrows()]
