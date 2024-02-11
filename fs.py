import pandas as pd
from pathlib import Path
from typing import List, Tuple, Iterable

from cloudchamber.particle import ParticleEvent

from config import ROOT_PATH

BG_RADIATION_PATH = ROOT_PATH / "Background"
ROD_RADIATION_PATH = ROOT_PATH / "Rod"

CSV_PATH = ROOT_PATH / "csv"

_COLUMNS = ("Width", "Length", "Angle",
            "Start Index", "Start Time",
            "End Index", "End Time")


def _is_video(path: Path) -> bool:
    return path.suffix.lower() == ".mp4"


def _get_videos(path: Path) -> List[Path]:
    return sorted(filter(_is_video, path.iterdir()))


def get_bg_videos() -> List[Path]:
    return _get_videos(BG_RADIATION_PATH)


def get_rod_videos() -> List[Path]:
    return _get_videos(ROD_RADIATION_PATH)


def _csv_row(particle: ParticleEvent) -> Tuple:
    return (particle.width, particle.length, particle.angle,
            particle.start.frame.index, particle.start.frame.timestamp.total_seconds(),
            particle.end.frame.index, particle.end.frame.timestamp.total_seconds())


def save_particles(particles: Iterable[ParticleEvent], path: Path) -> None:
    pd.DataFrame(map(_csv_row, particles), columns=_COLUMNS).to_csv(path)
