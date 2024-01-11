from numpy import ndarray
from itertools import islice

from fs import get_videos
from viewing import display_frame
from utils import monochrome, threshold
from parsing import parse_video, iter_frames, frame_num

THRESH = 20

BG_FRAME = 150
JUMP_FRAMES = 50


def process_frame(frame: ndarray) -> ndarray:
    return threshold(monochrome(frame), THRESH)


if __name__ == '__main__':
    example_path = list(get_videos())[1]
    print(f"Parsing {example_path}")
    with parse_video(example_path) as video:
        print(f"Video has {frame_num(video)} frames")
        frames = islice(iter_frames(video), BG_FRAME, None, JUMP_FRAMES)
        bg = process_frame(next(frames))
        display_frame(bg, "Background")
        for i, frame in enumerate(frames):
            title = f"Frame {i}"
            display_frame(process_frame(frame), title)
            display_frame(frame, title)
