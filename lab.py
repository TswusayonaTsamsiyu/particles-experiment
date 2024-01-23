import cv2
import numpy as np
from numpy import ndarray

import image as img
from viewing import display_frame
from fs import get_bg_videos, get_rod_videos
from parsing import parse_video, iter_frames, frame_num, next_frame_index

THRESH = 5

BG_FRAME = 3600
JUMP_FRAMES = 200

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
        frames = iter_frames(video, start=BG_FRAME, jump=JUMP_FRAMES)
        bg = (prepare(next(frames)))
        display_frame(bg, "Background")
        for frame in frames:
            title = f"Frame {next_frame_index(video) - 1}"
            prepared = prepare(frame)
            subtracted = cv2.subtract(prepared, bg)
            print(f"Max: {subtracted.max()}, Mean: {subtracted.mean()}, STD: {subtracted.std()}")
            if not has_tracks(subtracted):
                print(f"Frame {next_frame_index(video)} is the new BG")
                bg = prepare(frame)
            display_frame(prepared, title)
            display_frame(subtracted, title)
            display_frame(make_binary(subtracted), title)
