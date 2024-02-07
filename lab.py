from time import time
from pathlib import Path
from more_itertools import chunked_even
from typing import Sequence, List, MutableSequence, Iterable, Container, Callable, Generator

from bettercv.track import Track
from bettercv import image as img
from bettercv import display as disp
from bettercv.video import Video, Frame
from bettercv.contours import find_contours, draw_contours, join_close_contours, Contour

from particle import ParticleEvent
from fs import get_bg_videos, get_rod_videos

# Video analysis window
START_TIME = 120
NUM_SECONDS = 15

# Preprocessing
BLUR_SIZE = 15
KSIZE = (BLUR_SIZE, BLUR_SIZE)

# Handling keyboard interaction
ESC = 27
CLOSE_BTN = -1
EXIT_CODES = {ESC, CLOSE_BTN}

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


def exit_for(codes: Container[int]) -> Callable[[int], None]:
    return lambda key_code: exit() if key_code in codes else None


handle_key_code = exit_for(EXIT_CODES)


def preprocess(frame: Frame) -> Frame:
    return frame.with_image(img.blur(img.grayscale(frame.image), KSIZE))


def has_tracks(threshold: float) -> bool:
    return threshold > MIN_THRESHOLD


def find_prominent_contours(binary: Frame) -> Sequence[Contour]:
    return tuple(contour
                 for contour in find_contours(binary.image, external_only=True)
                 if contour.area() > MIN_CONTOUR_SIZE)


# def display_frame(frame: Frame, binary: Image, contours: Sequence[Contour]) -> None:
#     right_window = disp.Window(draw_contours(binary, contours),
#                                title=f"Binary {frame} with contours")
#     left_window = disp.Window(frame.image,
#                               title=f"Prepared {frame}",
#                               position=disp.left_of(right_window))
#     handle_key_code(disp.show(map(disp.fit_to_screen, (right_window, left_window))))


def find_close_tracks(contour: Contour, index: int, tracks: Iterable[Track]) -> List[Track]:
    return list(track for track in tracks
                if (track.end.contour.centroid().distance_to(contour.centroid()) < DRIFT_DISTANCE)
                and (index - track.end.index == 1))


def update_tracks(tracks: MutableSequence[Track],
                  contours: Sequence[Contour],
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


def display_particles(video: Video, events: Iterable[ParticleEvent]) -> None:
    for event in events:
        relevant_frame = video.read_frame_at(event.best_snapshot.index)
        handle_key_code(disp.show([disp.fit_to_screen(disp.Window(
            draw_contours(img.abc(relevant_frame.image), [event.best_snapshot.contour]),
            str(event.best_snapshot)
        ))]))


# def detect_tracks(video: Video, initial_bg: int, stop: int = None) -> List[Track]:
#     had_tracks = False
#     tracks: List[Track] = []
#     bg = prepare(video.read_frame_at(initial_bg).pixels)
#     for frame in video.iter_frames(start=initial_bg + 1, stop=stop):
#         # print(f"Processing {frame}")
#         thresh, binary = process_frame(frame, bg)
#         # print(f"Threshold: {thresh}")
#         if has_tracks(thresh):
#             had_tracks = True
#             # print("Tracks detected")
#             contours = find_tracks(binary)
#             update_tracks(tracks,
#                           join_close_contours(contours) if len(contours) > 1 else contours,
#                           frame, binary)
#         else:
#             # print("No tracks detected")
#             if had_tracks:
#                 print("Changing BG")
#                 bg = prepare(frame.pixels)
#             had_tracks = False
#     return tracks


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


def detect_tracks(frames: Iterable[Frame]) -> List[Track]:
    tracks: List[Track] = []
    for binary in binaries_with_tracks(subtract_bg(map(preprocess, frames))):
        contours = join_close_contours(find_prominent_contours(binary), DIST_CLOSE)
        update_tracks(tracks, contours, binary)
    return tracks


def analyze_video(path: Path) -> None:
    start = time()
    with Video(path) as video:
        print(f"Parsing {video.name}...")
        print(f"Video has {video.frame_num} frames.")
        tracks = detect_tracks(video.iter_frames(
            start=START_TIME * video.fps,
            stop=(START_TIME + NUM_SECONDS) * video.fps
        ))
        print(f"Num tracks found: {len(tracks)}")
        particle_events = [ParticleEvent(track)
                           for track in tracks
                           if track.extent > MIN_TRACK_LENGTH]
        print(f"Num particle events found: {len(particle_events)}")
        print(f"Finished in {time() - start} seconds.")
        display_particles(video, particle_events)


if __name__ == '__main__':
    analyze_video(list(get_bg_videos())[1])
