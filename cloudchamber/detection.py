from more_itertools import chunked_even
from typing import Iterable, Sequence, MutableSequence, List, Generator

import bettercv.image as img
from bettercv.track import Track
from bettercv.video import Frame
from bettercv.contours import Contour, find_contours, join_close_contours

from .particle import ParticleEvent

# Preprocessing
BLUR_SIZE = 15
KSIZE = (BLUR_SIZE, BLUR_SIZE)

# Tracking
DRIFT_DISTANCE = 40

# Joining
DIST_CLOSE = 100

# Filtering
MIN_CONTOUR_SIZE = 500
MIN_TRACK_LENGTH = 5
MIN_THRESHOLD = 1

# BG computation
BG_JUMP = 5
BG_BATCH_SIZE = 200

# Resizing
SCALE_FACTOR = 0.6


def preprocess(frame: Frame) -> Frame:
    return frame.with_image(img.blur(img.grayscale(img.scale(frame.image, SCALE_FACTOR)), KSIZE))


def has_tracks(threshold: float) -> bool:
    return threshold > MIN_THRESHOLD


def find_prominent_contours(binary: Frame) -> Sequence[Contour]:
    return tuple(contour
                 for contour in find_contours(binary.image, external_only=True)
                 if contour.area > MIN_CONTOUR_SIZE)


def retain_track_like(contours: Iterable[Contour]) -> Iterable[Contour]:
    return (contour for contour in contours
            if (contour.length / contour.width > 3)
            and (contour.width < 100))


def find_close_tracks(contour: Contour, index: int, tracks: Iterable[Track]) -> List[Track]:
    return list(track for track in tracks
                if (track.end.contour.centroid.distance_to(contour.centroid) < DRIFT_DISTANCE)
                and (index - track.end.frame.index == 1))


def update_tracks(tracks: MutableSequence[Track],
                  contours: Iterable[Contour],
                  binary: Frame) -> None:
    for contour in contours:
        close = find_close_tracks(contour, binary.index, tracks)
        if len(close) > 1:
            raise Exception("Multiple tracks detected for same contour!")
        if len(close) == 1:
            close[0].record(contour, binary)
        else:
            new_track = Track()
            new_track.record(contour, binary)
            tracks.append(new_track)


def subtract_bg(frames: Iterable[Frame]) -> Generator[Frame, None, None]:
    for batch in chunked_even(frames, BG_BATCH_SIZE):
        print(f"Computing BG for {batch[0].index}-{batch[-1].index}")
        bg = img.avg(frame.image for frame in batch[::BG_JUMP])
        # disp.show([disp.fit_to_screen(disp.Window(bg, "Avg BG"))])
        for frame in batch:
            yield frame.with_image(img.subtract(frame.image, bg))


def binaries_with_tracks(frames: Iterable[Frame]) -> Generator[Frame, None, None]:
    for frame in frames:
        thresh, binary = img.threshold_otsu(frame.image)
        if has_tracks(thresh):
            yield frame.with_image(binary)


def detect_tracks(frames: Iterable[Frame]) -> List[ParticleEvent]:
    tracks: List[Track] = []
    for binary in binaries_with_tracks(subtract_bg(map(preprocess, frames))):
        contours = retain_track_like(join_close_contours(find_prominent_contours(binary), DIST_CLOSE))
        update_tracks(tracks, contours, binary)
    return [ParticleEvent(track)
            for track in tracks
            if track.extent > MIN_TRACK_LENGTH]
