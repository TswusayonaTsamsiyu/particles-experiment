import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import namedtuple
from typing import List, Tuple, Iterable

from bettercv.video import Ref
from bettercv.track import Snapshot
from bettercv.contours import Contour

from cloudchamber.particle import Particle

from root import ROOT_PATH

BG_RADIATION_PATH = ROOT_PATH / "Background"
ROD_RADIATION_PATH = ROOT_PATH / "Rod"

CSV_PATH = ROOT_PATH / "csv"
GRAPH_PATH = ROOT_PATH / "graphs"

_COLUMNS = ("Width", "Length", "Angle", "Curvature", "Intensity", "Type",
            "StartIndex", "StartTime", "EndIndex", "EndTime",
            "SnapshotIndex", "SnapshotTime", "Video", "Contour")


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


def _serialize_ref(ref: Ref) -> Tuple[int, float]:
    return ref.index, ref.timestamp


def _serialize_particle(particle: Particle) -> Tuple:
    return (particle.width, particle.length, particle.angle, particle.curvature, particle.intensity, 0,
            *_serialize_ref(particle.start), *_serialize_ref(particle.end),
            *_serialize_ref(particle.snapshot.ref), particle.snapshot.ref.video,
            _serialize_contour(particle.snapshot.contour))


def _parse_particle(row: namedtuple) -> Particle:
    return Particle((Ref(row.Video, row.StartIndex, row.StartTime), Ref(row.Video, row.EndIndex, row.EndTime)),
                    Snapshot(Ref(row.Video, row.SnapshotIndex, row.SnapshotTime), _parse_contour(row.Contour)))


def save_particles(particles: Iterable[Particle], path: Path) -> None:
    data = pd.DataFrame(map(_serialize_particle, particles), columns=_COLUMNS)
    data.to_csv(path, index=False, mode="a", header=not path.exists())


def load_particles(path: Path) -> List[Particle]:
    return [_parse_particle(row) for row in pd.read_csv(path).itertuples()]
