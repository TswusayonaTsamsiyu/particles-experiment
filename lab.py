from time import time
from typing import Sequence, Tuple, List, MutableSequence, Iterable, Container, Callable

from bettercv.types import Image
from bettercv.track import Track
from bettercv import image as img
from bettercv import display as disp
from bettercv.video import Video, Frame
from bettercv.contours import find_contours, draw_contours, join_contours, Contour

from particle import ParticleEvent
from fs import get_bg_videos, get_rod_videos

BG_FRAME = 3600
NUM_SECONDS = 15

BLUR_SIZE = 15
KSIZE = (BLUR_SIZE, BLUR_SIZE)

ESC = 27
CLOSE_BTN = -1
EXIT_CODES = {ESC, CLOSE_BTN}

DRIFT_DISTANCE = 5

MIN_TRACK_LENGTH = 3


def exit_for(codes: Container[int]) -> Callable[[int], None]:
    return lambda key_code: exit() if key_code in codes else None


handle_key_code = exit_for(EXIT_CODES)


def prepare(frame: Image) -> Image:
    return img.blur(img.grayscale(frame), KSIZE)


def has_tracks(threshold: float) -> bool:
    return threshold > 1


def process_frame(frame: Frame, bg: Image) -> Tuple[float, Image]:
    # print(f"Processing frame {frame.index}")
    return img.threshold_otsu(img.subtract(prepare(frame.pixels), bg))


def find_tracks(binary: Image) -> Tuple[Contour]:
    return tuple(contour for contour in find_contours(binary, external_only=True) if contour.area() > 600)


def display_frame(frame: Frame, binary: Image, contours: Sequence[Contour]) -> None:
    right_window = disp.Window(draw_contours(binary, contours),
                               title=f"Binary {frame} with contours")
    left_window = disp.Window(frame.pixels,
                              title=f"Prepared {frame}",
                              position=disp.left_of(right_window))
    handle_key_code(disp.show(map(disp.fit_to_screen, (right_window, left_window))))


def find_close_tracks(contour: Contour, frame: Frame, tracks: Iterable[Track]) -> List[Track]:
    return list(track for track in tracks
                if (track.end.contour.centroid().distance_to(contour.centroid()) < DRIFT_DISTANCE)
                and (frame.index - track.end.index == 1))


def flatten_tree(tree: dict) -> Sequence[Sequence]:
    def flatten(tree: dict, key: int) -> List:
        flat = [key]
        if key in tree:
            for value in tree[key]:
                flat += flatten(tree, value)
        return flat

    return [flatten(tree, key) for key in tree]


def join_close_contours(contours: MutableSequence[Contour]) -> MutableSequence[Contour]:
    joined = set()
    to_join = dict()
    for i, c1 in enumerate(contours):
        for j, c2 in list(enumerate(contours))[i + 1:]:
            if j not in joined and c1.is_close_to(c2, 100):
                to_join.setdefault(i, []).append(j)
                joined.add(j)
    if to_join:
        print(to_join)
        print(flatten_tree(to_join))
    final = []
    for i, contour in enumerate(contours):
        if not (i in to_join or i in joined):
            final.append(contour)
    return final


def update_tracks(tracks: MutableSequence[Track],
                  contours: Sequence[Contour],
                  frame: Frame,
                  binary: Image) -> None:
    for contour in contours:
        close = find_close_tracks(contour, frame, tracks)
        if len(close) > 1:
            display_frame(frame, binary, [track.end.contour for track in close])
            raise Exception("Multiple tracks detected for same contour!")
        if len(close) == 1:
            close[0].record(contour, frame)
        else:
            new_track = Track()
            new_track.record(contour, frame)
            tracks.append(new_track)
            # display_frame(frame, binary, contours)


def display_particles(video: Video, events: Iterable[ParticleEvent]) -> None:
    for event in events:
        relevant_frame = video.read_frame_at(event.best_snapshot.index)
        handle_key_code(disp.show([disp.fit_to_screen(disp.Window(
            draw_contours(relevant_frame.pixels, [event.best_snapshot.contour]),
            str(event.best_snapshot)
        ))]))


def detect_tracks(video: Video, initial_bg: int, stop: int = None) -> List[Track]:
    had_tracks = False
    tracks: List[Track] = []
    bg = prepare(video.read_frame_at(initial_bg).pixels)
    for frame in video.iter_frames(start=initial_bg + 1, stop=stop):
        # print(f"Processing {frame}")
        thresh, binary = process_frame(frame, bg)
        # print(f"Threshold: {thresh}")
        if has_tracks(thresh):
            had_tracks = True
            # print("Tracks detected")
            contours = find_tracks(binary)
            if len(contours) > 1:
                join_close_contours(contours)
            update_tracks(tracks, contours, frame, binary)
        else:
            # print("No tracks detected")
            if had_tracks:
                print("Changing BG")
                bg = prepare(frame.pixels)
            had_tracks = False
    return tracks


def main() -> None:
    start = time()
    example_path = list(get_bg_videos())[1]
    print(f"Parsing {example_path.name}...")
    with Video(example_path) as video:
        print(f"Video has {video.frame_num} frames.")
        tracks = detect_tracks(video, BG_FRAME, BG_FRAME + NUM_SECONDS * video.fps)
        print(f"Num tracks found: {len(tracks)}")
        particle_events = [ParticleEvent(track)
                           for track in tracks
                           if track.extent > MIN_TRACK_LENGTH]
        print(f"Num particle events found: {len(particle_events)}")
        print(f"Finished in {time() - start} seconds.")
        display_particles(video, particle_events)


if __name__ == '__main__':
    main()
