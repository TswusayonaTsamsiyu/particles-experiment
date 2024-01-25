import cv2 as cv
import numpy as np
from numpy import ndarray
from typing import Iterable
from functools import reduce

import image as img
from video import Video, Frame
from viewing import window_control, show_window, Position
from fs import get_bg_videos, get_rod_videos

BG_FRAME = 3600
JUMP_FRAMES = 20

BLUR_SIZE = 15
KSIZE = (BLUR_SIZE, BLUR_SIZE)


# TODO:
# 1. Find local BG
# 2. Find auto threshold


def prepare(frame: ndarray) -> ndarray:
    return img.blur(img.monochrome(frame), KSIZE)


def make_binary(frame: ndarray) -> ndarray:
    return img.threshold(frame, np.std(frame) * 5)
    # return img.blur(img.threshold(img.adjust_brightness_contrast(frame)), KSIZE)


def has_tracks(frame: ndarray) -> bool:
    mean, std = cv.meanStdDev(frame)
    min_, max_ = cv.minMaxLoc(frame)[:2]
    print(f"Mean: {mean[0][0]}, STD: {std[0][0]}, Max: {max_}")
    return mean > 0.6 and std > 1 and max_ > 25


def get_avg_bg(frames: Iterable[ndarray], window: int) -> ndarray:
    return cv.divide(window, reduce(cv.add, map(prepare, frames)))


def analyze_frame(frame: Frame, bg: ndarray) -> None:
    print(f"Processing frame {frame.index}")
    prepared = prepare(frame.pixels)
    with window_control():
        show_window(prepared,
                    title=f"Prepared frame {frame.index}",
                    position=Position(0, 0))
        subtracted = cv.subtract(prepared, bg)
        print("Tracks detected" if has_tracks(subtracted) else "No tracks")
        show_window(make_binary(subtracted),
                    title=f"Binary frame {frame.index}",
                    position=Position(600, 0))


def analyze_video(video: Video) -> None:
    frames = video.iter_frames(start=BG_FRAME, jump=JUMP_FRAMES)
    bg = prepare(next(frames).pixels)
    with window_control():
        show_window(bg, f"Background frame {BG_FRAME}")
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
