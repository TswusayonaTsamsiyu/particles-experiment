from typing import Sequence, Tuple, List, MutableSequence

from bettercv import image as img
from bettercv import display as disp
from bettercv.video import Video, Frame
from bettercv.utils import Image, Position, exit_for
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
    with disp.window_control(exit_for(EXIT_CODES)):
        shown = disp.fit_to_screen(frame.pixels)
        prepwin = disp.show_window(shown,
                                   title=f"Prepared frame {frame.index}",
                                   position=Position(disp.screen_center().x - shown.shape[1] - 200, 0))
        disp.show_window(disp.fit_to_screen(draw_contours(binary, contours)),
                         title=f"Binary frame {frame.index} with contours",
                         position=disp.right_of(prepwin))


def distance(point1: Position, point2: Position) -> float:
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** .5


def update_tracks(tracks: MutableSequence[Track], contours: Sequence[Contour], frame: Frame, binary: Image) -> None:
    for contour in contours:
        close = list(track for track in tracks
                     if (distance(track.contours[-1].centroid(), contour.centroid()) < DRIFT_DISTANCE)
                     and (frame.index - track.end.index == 1))
        if len(close) > 1:
            binary_with_tracks = draw_contours(binary, [track.contours[-1] for track in close])
            display_frame(frame, binary_with_tracks, contours)
            raise Exception("Multiple tracks detected for same contour!")
        if len(close) == 1:
            close[0].append(contour)
            close[0].end = frame
        else:
            tracks.append(Track([contour], frame))
            # display_frame(frame, binary, contours)

def display_tracks(video: Video, tracks: MutableSequence[Track], start: int, stop: int = None) -> None:
    bg = prepare(video.read_frame_at(start).pixels)
    frame_indexes = [track.relevant_frame_index() for track in tracks]
    for frame in video.iter_frames(start=start + 1, stop=stop):
        # print(f"Looking for tracks in frame {frame.index}")
        if frame.index in frame_indexes:
            for track in tracks:
                if track.relevant_frame_index() == frame.index:
                    thresh, binary = process_frame(frame, bg)
                    print(f"Frame timestamp: {frame.timestamp}")
                    display_frame(frame, binary, [track.relevant_contour()])

def analyze_video(video: Video, start: int, stop: int = None) -> List[Track]:
    tracks: List[Track] = []
    had_tracks = False
    bg = prepare(video.read_frame_at(start).pixels)
    for frame in video.iter_frames(start=start + 1, stop=stop):
        # print(f"Frame timestamp: {frame.timestamp}")
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
        tracks = analyze_video(video, BG_FRAME, NUM_SECONDS*video.fps)
        print(f"Num tracks: {len(tracks)}")
        relevant_tracks = [track for track in tracks if track.length() > MIN_TRACK_LENGTH]
        print(f"Num relevant tracks: {len(relevant_tracks)}")
        relevant_frame_indexes = [track.relevant_frame_index() for track in relevant_tracks]
        print(f"Relevant_frame_indexes: {relevant_frame_indexes}")
        display_tracks(video, relevant_tracks, BG_FRAME, NUM_SECONDS*video.fps)
        print(f"Finished")


if __name__ == '__main__':
    main()
