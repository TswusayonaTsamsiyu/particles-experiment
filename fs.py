from pathlib import Path
from typing import Iterable

from config import ROOT_PATH

BG_RADIATION_PATH = ROOT_PATH / "Background"
ROD_RADIATION_PATH = ROOT_PATH / "Rod"


def is_video(path: Path) -> bool:
    return path.suffix.lower() == ".mp4"


def get_videos(path: Path) -> Iterable[Path]:
    return sorted(filter(is_video, path.iterdir()))


def get_bg_videos() -> Iterable[Path]:
    return get_videos(BG_RADIATION_PATH)


def get_rod_videos() -> Iterable[Path]:
    return get_videos(ROD_RADIATION_PATH)
