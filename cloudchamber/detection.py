import numpy as np
from pathlib import Path
from typing import Iterable, Sequence, MutableSequence, List

from bettercv.track import Track
from bettercv.video import Video, Frame
from bettercv.contours import Contour, find_contours, join_close_contours

import cloudchamber.debugging as dbg

from .config import Config
from .particle import Particle
from .processing import preprocess, smooth
from .bg_subtraction import subtract_bg, subtract_bg_avg, binaries_with_tracks_2


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


def angle_between_contours(contour1: Contour, contour2: Contour) -> float:
    center1, center2 = contour1.centroid, contour2.centroid
    angle = np.rad2deg(np.arctan((center1.y - center2.y) / (center1.x - center2.x)))
    return angle if angle > 0 else angle + 180


def is_drift(contour: Contour, track: Track) -> bool:
    return 10 < abs(angle_between_contours(contour, track.end.contour) - track.end.contour.angle) < 170


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


def is_track_drift(track1: Track, track2: Track) -> bool:
    angle = angle_between_contours(track1.end.contour, track2.start.contour)
    return ((track2.start.ref.index - track1.end.ref.index < 3)
            and (abs(track1.end.contour.angle - track2.start.contour.angle) < 20)
            and (70 < angle < 110))


def _filter_tracks(tracks: Sequence[Track], config: Config) -> List[Track]:
    filtered = [track for track in tracks if track.extent > config.min_track_length]
    for track1 in filtered:
        for track2 in filtered:
            if is_track_drift(track1, track2):
                filtered.remove(track2)
    return filtered
    # if (abs(track.end.contour.angle - contour.angle) < 20)
    # and (abs(track.end.contour.width - contour.width) < 20)]


def _filter_tracks2(tracks: List[Track], config: Config) -> Sequence[Track]:
    for track1 in tracks:
        for track2 in tracks:
            if is_track_drift(track1, track2):
                tracks.remove(track2)
    return [track for track in tracks if track.extent > config.min_track_length]


def filter_tracks(tracks: List[Track], config: Config) -> List[Track]:
    return [track for track in tracks if track.extent > config.min_track_length]


def detect_tracks(frames: Iterable[Frame], **config) -> List[Particle]:
    tracks: List[Track] = []
    config = Config.merge(config)
    # frames = (smooth(preprocess(frame, config), config) for frame in frames)
    # frames = binaries_with_tracks_2(subtract_bg_avg(frames, config), config)
    frames = subtract_bg((smooth(preprocess(frame, config), config) for frame in frames), config)
    for binary in frames:
        contours = list(retain_track_like(
            join_close_contours(
                find_prominent_contours(binary, config.min_contour_size),
                config.dist_close
            ), config
        ))
        if contours:
            dbg.display_contours(binary, contours)
        update_tracks(tracks, contours, binary, config)
    # display_particles(all_particles)
    return list(map(
        Particle.from_track,
        filter_tracks(tracks, config)
    ))
    # maybe check growth rate
    # and track.start.contour.length < 50]


def analyze_video(path: Path, start: int = 0, stop: int = None, **config) -> List[Particle]:
    with Video(path) as video:
        return detect_tracks(video.iter_frames(
            start=video.index_at(start),
            stop=video.index_at(stop) if stop else None
        ), **config)
