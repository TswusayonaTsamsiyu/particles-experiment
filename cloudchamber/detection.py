from pathlib import Path
from more_itertools import chunked
from typing import Iterable, Sequence, MutableSequence, List, Generator

import bettercv.image as img
from bettercv.track import Track
from bettercv.display import Window
from bettercv.video import Video, Frame
from bettercv.contours import Contour, find_contours, join_close_contours

from .config import Config
from .particle import Particle


def preprocess(frame: Frame, config: Config) -> Frame:
    return frame.with_image(img.blur(
        img.grayscale(img.scale(frame.image, config.scale_factor)),
        (config.blur_size, config.blur_size)
    ))


def has_tracks(threshold: float, min_thresh: float) -> bool:
    return threshold > min_thresh


def find_prominent_contours(binary: Frame, min_size: int) -> Sequence[Contour]:
    return tuple(contour
                 for contour in find_contours(binary.image, external_only=True)
                 if contour.area > min_size)


def retain_track_like(contours: Iterable[Contour]) -> Iterable[Contour]:
    return (contour for contour in contours
            if (contour.length / contour.width > 3)
            and (contour.width < 100))


def find_close_tracks(contour: Contour, index: int, tracks: Iterable[Track], drift_distance: int) -> List[Track]:
    return list(track for track in tracks
                if (track.end.contour.centroid.distance_to(contour.centroid) < drift_distance)
                and (index - track.end.frame.index == 1))


def update_tracks(tracks: MutableSequence[Track],
                  contours: Iterable[Contour],
                  binary: Frame,
                  config: Config) -> None:
    for contour in contours:
        close = find_close_tracks(contour, binary.index, tracks, config.drift_distance)
        if len(close) > 1:
            raise Exception("Multiple tracks detected for same contour!")
        if len(close) == 1:
            close[0].record(contour, binary)
        else:
            new_track = Track()
            new_track.record(contour, binary)
            tracks.append(new_track)


def subtract_bg(frames: Iterable[Frame], config: Config) -> Generator[Frame, None, None]:
    for batch in chunked(frames, config.bg_batch_size):
        if config.prints:
            print(f"Computing BG for {batch[0].index}-{batch[-1].index}")
        bg = img.avg(frame.image for frame in batch[::config.bg_jump])
        if config.display:
            Window(bg, "Avg BG").fit_to_screen().show()
        for frame in batch:
            yield frame.with_image(img.subtract(frame.image, bg))


def binaries_with_tracks(frames: Iterable[Frame], config: Config) -> Generator[Frame, None, None]:
    for frame in frames:
        thresh, binary = img.threshold_otsu(frame.image)
        if has_tracks(thresh, config.min_threshold):
            yield frame.with_image(binary)


def detect_tracks(frames: Iterable[Frame], **config) -> List[Particle]:
    tracks: List[Track] = []
    config = Config.merge(config)
    for binary in binaries_with_tracks(subtract_bg((preprocess(frame, config) for frame in frames), config), config):
        contours = retain_track_like(
            join_close_contours(find_prominent_contours(binary, config.min_contour_size), config.dist_close))
        update_tracks(tracks, contours, binary, config)
    return [Particle.from_track(track)
            for track in tracks
            if track.extent > config.min_track_length]


def analyze_video(path: Path, start: int = 0, stop: int = None, **config) -> List[Particle]:
    with Video(path) as video:
        return detect_tracks(video.iter_frames(
            start=video.index_at(start),
            stop=video.index_at(stop)
        ), **config)
