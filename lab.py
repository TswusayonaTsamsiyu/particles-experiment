import cv2 as cv
import numpy as np

import image as img
import display as disp
from video import Video, Frame
from utils import Image, Position, exit_for
from fs import get_bg_videos, get_rod_videos

BG_FRAME = 3600
JUMP_FRAMES = 20

BLUR_SIZE = 15
KSIZE = (BLUR_SIZE, BLUR_SIZE)

ESC = 27
CLOSE_BTN = -1
EXIT_CODES = {ESC, CLOSE_BTN}


# TODO:
# 1. Find local BG
# 2. Find auto threshold


def prepare(frame: Image) -> Image:
    return img.blur(img.monochrome(frame), KSIZE)


def make_binary(frame: Image) -> Image:
    return img.threshold(frame, np.std(frame) * 5)


def has_tracks(frame: Image) -> bool:
    mean, std = cv.meanStdDev(frame)
    min_, max_ = cv.minMaxLoc(frame)[:2]
    print(f"Mean: {round(float(mean[0][0]), 3)}, STD: {round(float(std[0][0]), 3)}, Max: {max_}")
    return mean > 0.6 and std > 1 and max_ > 25


def analyze_frame(frame: Frame, bg: Image) -> None:
    print(f"Processing frame {frame.index}")
    prepared = prepare(frame.pixels)
    subtracted = cv.subtract(prepared, bg)
    binary = make_binary(subtracted)
    print("Tracks detected" if has_tracks(subtracted) else "No tracks")
    contours = img.find_contours(binary)
    with disp.window_control(exit_for(EXIT_CODES)):
        prepwin = disp.show_window(disp.fit_to_screen(prepared),
                                   title=f"Prepared frame {frame.index}",
                                   position=Position(0, 0))
        disp.show_window(disp.fit_to_screen(img.draw_contours(binary, contours)),
                         title=f"Binary frame {frame.index} with contours",
                         position=disp.right_of(prepwin))


def analyze_video(video: Video) -> None:
    frames = video.iter_frames(start=BG_FRAME, jump=JUMP_FRAMES)
    bg = prepare(next(frames).pixels)
    disp.show_single_window(disp.fit_to_screen(bg),
                            title=f"Background frame {BG_FRAME}",
                            position=Position(0, 0))
    for frame in frames:
        analyze_frame(frame, bg)


def main() -> None:
    example_path = list(get_bg_videos())[1]
    print(f"Parsing {example_path.name}...")
    with Video(example_path) as video:
        print(f"Video has {video.frame_num} frames.")
        analyze_video(video)


if __name__ == '__main__':
    main()
