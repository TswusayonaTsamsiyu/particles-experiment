from pathlib import Path
from typing import Iterable

LAB_PATH = Path("H:\\My Drive\\Labs\\Physics Lab C")
VIDEO_SUFFIX = ".mp4"


def is_video(path: Path) -> bool:
    return path.suffix == VIDEO_SUFFIX


def get_videos() -> Iterable[Path]:
    return filter(is_video, LAB_PATH.iterdir())
