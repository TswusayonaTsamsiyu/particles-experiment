from itertools import islice

from fs import get_videos
from parsing import get_frames
from viewing import display_frame
from utils import monochrome, threshold

if __name__ == '__main__':
    # print(frame_num((list(get_videos())[1])))
    frames = islice(get_frames(list(get_videos())[1]), 150, None, 50)
    bg = threshold(monochrome(next(frames)), 20)
    display_frame(bg)
    for frame in frames:
        display_frame(threshold(monochrome(frame), 20))
        display_frame(frame)
