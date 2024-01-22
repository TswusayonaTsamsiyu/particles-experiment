import cv2
import numpy as np
from numpy import ndarray
from itertools import islice

import image as img
from viewing import display_frame
from fs import get_bg_videos, get_rod_videos
from parsing import parse_video, iter_frames, frame_num, frame_index

THRESH = 5

BG_FRAME = 150
JUMP_FRAMES = 50

BLUR_SIZE = 15
KSIZE = (BLUR_SIZE, BLUR_SIZE)


# TODO:
# 1. Find local BG
# 2. Find auto threshold (maybe find max brightness of photo )


def prepare(frame: ndarray) -> ndarray:
    return img.blur(img.monochrome(frame), KSIZE)


def make_binary(frame: ndarray) -> ndarray:
    return img.threshold(frame, np.std(frame) * 5)
    # return img.blur(img.threshold(img.adjust_brightness_contrast(frame)), KSIZE)


def has_tracks(frame: ndarray) -> bool:
    return frame.mean() > 0.2


if __name__ == '__main__':
    example_path = list(get_bg_videos())[1]
    print(f"Parsing {example_path}")
    with parse_video(example_path) as video:
        print(f"Video has {frame_num(video)} frames")
        frames = islice(iter_frames(video), BG_FRAME, None, JUMP_FRAMES)
        bg = (prepare(next(frames)))
        display_frame(bg, "Background")
        for frame in frames:
            title = f"Frame {frame_index(video)}"
            prepared = prepare(frame)
            subtracted = cv2.subtract(prepared, bg)
            print(f"Max: {subtracted.max()}, Mean: {subtracted.mean()}, STD: {subtracted.std()}")
            if not has_tracks(subtracted):
                print(f"Frame {frame_index(video)} is the new BG")
                bg = prepare(frame)
            display_frame(subtracted)
            display_frame(make_binary(subtracted), title)
            display_frame(prepared, title)
