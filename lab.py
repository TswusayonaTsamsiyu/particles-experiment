import cv2
import numpy as np
from numpy import ndarray
from typing import Iterable
from functools import reduce

import image as img
from viewing import display_frame
from fs import get_bg_videos, get_rod_videos
from parsing import parse_video, iter_frames, frame_num, next_frame_index

THRESH = 5

BG_FRAME = 3600
JUMP_FRAMES = 20

BLUR_SIZE = 15
KSIZE = (BLUR_SIZE, BLUR_SIZE)

WINDOW = 200  # frames


# TODO:
# 1. Find local BG
# 2. Find auto threshold (maybe find max brightness of photo )


def prepare(frame: ndarray) -> ndarray:
    return img.blur(img.monochrome(frame), KSIZE)


def make_binary(frame: ndarray) -> ndarray:
    return img.threshold(frame, np.std(frame) * 5)
    # return img.blur(img.threshold(img.adjust_brightness_contrast(frame)), KSIZE)


def has_tracks(frame: ndarray) -> bool:
    mean, std = cv2.meanStdDev(subtracted)
    min, max = cv2.minMaxLoc(subtracted)[:2]
    return mean > 0.6 and std > 1 and max > 25


def get_avg_bg(frames: Iterable[ndarray], window: int) -> ndarray:
    return cv2.divide(window, reduce(cv2.add, map(prepare, frames)))


if __name__ == '__main__':
    example_path = list(get_bg_videos())[1]
    print(f"Parsing {example_path}")
    with parse_video(example_path) as video:
        print(f"Video has {frame_num(video)} frames")
        frames = iter_frames(video, start=BG_FRAME, jump=JUMP_FRAMES)
        bg = prepare(next(frames))
        # bg = get_avg_bg(iter_frames(video, start=BG_FRAME, stop=BG_FRAME + WINDOW), WINDOW)
        display_frame(bg, "Background")
        for frame in frames:
            index = next_frame_index(video) - 1
            title = f"Frame {index}"
            prepared = prepare(frame)
            subtracted = cv2.subtract(prepared, bg)
            print(f"Mean, STD: {cv2.meanStdDev(subtracted)}\nMin, Max: {cv2.minMaxLoc(subtracted)[:2]}")
            if has_tracks(subtracted):
                # print(f"Frame {index} is the new BG")
                # bg = prepared
                # bg = get_avg_bg(iter_frames(video, start=index - WINDOW, stop=index + WINDOW), WINDOW * 2)
                print("Tracks detected")
                display_frame(prepared, title)
                display_frame(make_binary(subtracted), title)
            else:
                print("No tracks")
                display_frame(prepared, title)
                display_frame(make_binary(subtracted), title)

