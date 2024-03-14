from pathlib import Path
from typing import Iterable, Sequence, MutableSequence, List

from bettercv.track import Track
from bettercv.video import Video, Frame
from bettercv.contours import Contour, find_contours, join_close_contours

import cloudchamber.debugging as dbg

from .config import Config
from .particle import Particle
from .bg_subtraction import subtract_bg
from .processing import preprocess, smooth


def find_prominent_contours(binary: Frame, min_size: int) -> Sequence[Contour]:
    return tuple(contour
                 for contour in find_contours(binary.image, external_only=True)
                 if contour.area > min_size)


def retain_track_like(contours: Iterable[Contour], config: Config) -> Iterable[Contour]:
    return (contour for contour in contours
            if (contour.length / contour.width > config.min_aspect_ratio)
            and (contour.width < config.max_contour_width))


def find_close_tracks(contour: Contour, index: int, tracks: Iterable[Track], track_distance: int) -> List[Track]:
    return list(track for track in tracks
                if (track.end.contour.centroid.distance_to(contour.centroid) < track_distance)
                and (index - track.end.ref.index == 1))


def update_tracks(tracks: MutableSequence[Track],
                  contours: Iterable[Contour],
                  binary: Frame,
                  config: Config) -> None:
    for contour in contours:
        close = find_close_tracks(contour, binary.ref.index, tracks, config.track_distance)
        if len(close) > 1:
            # raise Exception("Multiple tracks detected for same contour!")
            pass
        if len(close) == 1:
            close[0].record(contour, binary)
        else:
            new_track = Track()
            new_track.record(contour, binary)
            tracks.append(new_track)


def detect_tracks(frames: Iterable[Frame], **config) -> List[Particle]:
    tracks: List[Track] = []
    config = Config.merge(config)
    frames = subtract_bg((smooth(preprocess(frame, config), config) for frame in frames), config)
    for binary in frames:
        contours = retain_track_like(
            join_close_contours(
                find_prominent_contours(binary, config.min_contour_size),
                config.dist_close
            ), config
        )
        update_tracks(tracks, contours, binary, config)
    return list(map(
        Particle.from_track,
        [track for track in tracks if track.extent > config.min_track_length]
    ))


def analyze_video(path: Path, start: int = 0, stop: int = None, **config) -> List[Particle]:
    with Video(path) as video:
        return detect_tracks(video.iter_frames(
            start=video.index_at(start),
            stop=video.index_at(stop) if stop else None
        ), **config)
