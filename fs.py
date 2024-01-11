from pathlib import Path
from typing import Iterable

from config import ROOT_PATH


def is_video(path: Path) -> bool:
    return path.suffix == ".mp4"


def get_videos() -> Iterable[Path]:
    return filter(is_video, ROOT_PATH.iterdir())
