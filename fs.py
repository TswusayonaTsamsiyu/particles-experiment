import json
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import ExitStack
from collections import namedtuple
from typing import List, Tuple, Iterable

from bettercv.video import Video
from bettercv.track import Snapshot
from bettercv.contours import Contour

from cloudchamber.config import Config
from cloudchamber.detection import preprocess
from cloudchamber.particle import ParticleTrack

from root import ROOT_PATH

BG_RADIATION_PATH = ROOT_PATH / "Background"
ROD_RADIATION_PATH = ROOT_PATH / "Rod"

CSV_PATH = ROOT_PATH / "csv"

_COLUMNS = ("Width", "Length", "Angle",
            "Start", "End",
            "SnapshotIndex", "Contour",
            "Video")


def _is_video(path: Path) -> bool:
    return path.suffix.lower() == ".mp4"


def _get_videos(path: Path) -> List[Path]:
    return sorted(filter(_is_video, path.iterdir()))


def get_bg_videos() -> List[Path]:
    return _get_videos(BG_RADIATION_PATH)


def get_rod_videos() -> List[Path]:
    return _get_videos(ROD_RADIATION_PATH)


def get_csvs() -> List[Path]:
    return [path for path in CSV_PATH.iterdir() if path.suffix.lower() == ".csv"]


def _serialize_contour(contour: Contour) -> str:
    return json.dumps(contour.points.tolist())


def _parse_contour(points: str) -> Contour:
    return Contour(np.array(json.loads(points), dtype=np.int32))


def _serialize_particle(particle: ParticleTrack) -> Tuple:
    return (particle.width, particle.length, particle.angle,
            particle.start, particle.end,
            particle.snapshot.frame.index, _serialize_contour(particle.snapshot.contour),
            particle.snapshot.frame.video)


def _parse_particle(row: namedtuple, video: Video) -> ParticleTrack:
    return ParticleTrack(
        (row.Start, row.End),
        Snapshot(
            preprocess(video[row.SnapshotIndex], Config),
            _parse_contour(row.Contour)
        )
    )


def save_particles(particles: Iterable[ParticleTrack], path: Path) -> None:
    pd.DataFrame(map(_serialize_particle, particles), columns=_COLUMNS).to_csv(path, index=False)


def read_particles(path: Path) -> List[ParticleTrack]:
    df = pd.read_csv(path)
    with ExitStack() as stack:
        videos = {path: stack.enter_context(Video(path)) for path in df.Video.unique()}
        return [_parse_particle(row, videos[row.Video]) for row in df.itertuples()]
