from numpy import ndarray
from itertools import islice

from viewing import display_frame
from fs import get_bg_videos, get_rod_videos
from image import monochrome, threshold, subtract_bg, crop, blur
from parsing import parse_video, iter_frames, frame_num, frame_index

THRESH = 30

BG_FRAME = 150
JUMP_FRAMES = 50


def process_frame(frame: ndarray) -> ndarray:
    # return subtract_bg(monochrome(frame), THRESH)
    return threshold(blur(monochrome(crop(frame, 400, 20)), (15, 15)), THRESH)


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
            display_frame(process_frame(frame), title)
            display_frame(frame, title)
