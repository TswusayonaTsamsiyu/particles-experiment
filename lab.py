from typing import Sequence, Tuple, List, MutableSequence, Iterable

from bettercv import image as img
from bettercv import display as disp
from bettercv.video import Video, Frame
from bettercv.utils import Image, exit_for, distance
from bettercv.contours import find_contours, draw_contours, Contour

from track import Track
from fs import get_bg_videos, get_rod_videos

BG_FRAME = 3600
JUMP_FRAMES = 20
END_FRAME = BG_FRAME + 450
NUM_SECONDS = 15

BLUR_SIZE = 15
KSIZE = (BLUR_SIZE, BLUR_SIZE)

ESC = 27
CLOSE_BTN = -1
EXIT_CODES = {ESC, CLOSE_BTN}

DRIFT_DISTANCE = 5

MIN_TRACK_LENGTH = 3

handle_key_code = exit_for(EXIT_CODES)


def prepare(frame: Image) -> Image:
    return img.blur(img.grayscale(frame), KSIZE)


# def make_binary(frame: Image) -> Image:
# return img.adaptive_threshold(frame, adaptive_method=cv.ADAPTIVE_THRESH_GAUSSIAN_C ,block_size=71 ,cut=1)
# return img.threshold(frame, np.std(frame) * 5)


# def has_tracks(frame: Image) -> bool:
#     mean, std = cv.meanStdDev(frame)
#     min_, max_ = cv.minMaxLoc(frame)[:2]
#     print(f"Mean: {round(float(mean[0][0]), 3)}, STD: {round(float(std[0][0]), 3)}, Max: {max_}")
#     return mean > 0.6 and std > 1 and max_ > 25


def has_tracks(threshold: float) -> bool:
    return threshold > 1


def process_frame(frame: Frame, bg: Image) -> Tuple[float, Image]:
    # print(f"Processing frame {frame.index}")
    return img.threshold_otsu(img.subtract(prepare(frame.pixels), bg))


def find_tracks(binary: Image) -> Sequence[Contour]:
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
                if (distance(track.end.contour.centroid(), contour.centroid()) < DRIFT_DISTANCE)
                and (frame.index - track.end.index == 1))


def update_tracks(tracks: MutableSequence[Track],
                  contours: Sequence[Contour],
                  frame: Frame, 
                  binary: Image) -> None:
    for contour in contours:
        close = find_close_tracks(contour, frame, tracks)
        if len(close) > 1:
            binary_with_tracks = draw_contours(binary, [track.end.contour for track in close])
            display_frame(frame, binary_with_tracks, contours)
            raise Exception("Multiple tracks detected for same contour!")
        if len(close) == 1:
            close[0].record(contour, frame)
        else:
            new_track = Track()
            new_track.record(contour, frame)
            tracks.append(new_track)
            # display_frame(frame, binary, contours)


def display_tracks(video: Video, tracks: Iterable[Track]) -> None:
    for track in tracks:
        relevant_frame = video.read_frame_at(track.relevant_snapshot.index)
        handle_key_code(disp.show([disp.fit_to_screen(disp.Window(
            draw_contours(relevant_frame.pixels, [track.relevant_snapshot.contour]),
            str(track.relevant_snapshot)
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
            update_tracks(tracks, contours, frame, binary)
        else:
            # print("No tracks detected")
            if had_tracks:
                print("Changing BG")
                bg = prepare(frame.pixels)
            had_tracks = False
    return tracks


def main() -> None:
    example_path = list(get_bg_videos())[1]
    print(f"Parsing {example_path.name}...")
    with Video(example_path) as video:
        print(f"Video has {video.frame_num} frames.")
        bg_frame = BG_FRAME  # 20240109_122031.mp4, list(get_bg_videos())[1]
        # bg_frame = (17 * 60 + 50) * video.fps # MAH00530.MP4, list(get_bg_videos())[3]
        stop = bg_frame + NUM_SECONDS * video.fps
        tracks = detect_tracks(video, bg_frame, stop)
        print(f"Num tracks found: {len(tracks)}")
        relevant_tracks = [track for track in tracks if track.extent > MIN_TRACK_LENGTH]
        print(f"Num relevant tracks found (len > {MIN_TRACK_LENGTH}): {len(relevant_tracks)}")
        display_tracks(video, relevant_tracks)
        print(f"Finished")


if __name__ == '__main__':
    main()
