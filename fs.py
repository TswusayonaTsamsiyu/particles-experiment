import pandas as pd
from pathlib import Path
from typing import List, Iterable

from bettercv.video import Video

from config import ROOT_PATH
from particle import ParticleEvent

BG_RADIATION_PATH = ROOT_PATH / "Background"
ROD_RADIATION_PATH = ROOT_PATH / "Rod"

CSV_PATH = ROOT_PATH / "csv"


def is_video(path: Path) -> bool:
    return path.suffix.lower() == ".mp4"


def get_videos(path: Path) -> List[Path]:
    return sorted(filter(is_video, path.iterdir()))


def get_bg_videos() -> List[Path]:
    return get_videos(BG_RADIATION_PATH)


def get_rod_videos() -> List[Path]:
    return get_videos(ROD_RADIATION_PATH)


def save_particles(particles: Iterable[ParticleEvent], video: Video) -> None:
    (pd.DataFrame([(particle.width, particle.length,
                    particle.start.index, particle.start.timestamp.total_seconds(),
                    particle.end.index, particle.end.timestamp.total_seconds())
                   for particle in particles],
                  columns=("Width", "Length", "Start Index", "Start Time", "End Index", "End Time"))
     .to_csv(CSV_PATH / video.path.with_suffix(".csv").name))
