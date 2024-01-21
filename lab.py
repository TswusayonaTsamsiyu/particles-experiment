from numpy import ndarray
from itertools import islice

import image as img
from viewing import display_frame
from fs import get_bg_videos, get_rod_videos
from parsing import parse_video, iter_frames, frame_num, frame_index

THRESH = 180

BG_FRAME = 150
JUMP_FRAMES = 50

BLUR_SIZE = 11
KSIZE = (BLUR_SIZE, BLUR_SIZE)


def process_frame(frame: ndarray) -> ndarray:
    return img.threshold(img.adjust_brightness_contrast(img.blur(
        img.monochrome(img.crop(frame, 400, 20)), KSIZE
    )), THRESH)


if __name__ == '__main__':
    example_path = list(get_bg_videos())[1]
    print(f"Parsing {example_path}")
    with parse_video(example_path) as video:
        print(f"Video has {frame_num(video)} frames")
        frames = islice(iter_frames(video), BG_FRAME, None, JUMP_FRAMES)
        bg = process_frame(next(frames))
        display_frame(bg, "Background")
        for frame in frames:
            title = f"Frame {frame_index(video)}"
            display_frame(process_frame(frame) - bg, title)
            display_frame(img.crop(frame, 400, 20), title)
