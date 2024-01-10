from pathlib import Path
from typing import Generator
from itertools import islice

import cv2
import numpy

LAB_PATH = Path("H:\\My Drive\\Labs\\Physics Lab C")
VIDEO_SUFFIX = ".mp4"
TITLE = "Frame"
RESIZE_FACTOR = 0.4


def is_video(path: Path) -> bool:
    return path.suffix == VIDEO_SUFFIX


def get_videos():
    return filter(is_video, LAB_PATH.iterdir())


def get_frames(path: Path):
    video = cv2.VideoCapture(str(path))

    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame

    video.release()


def frame_num(path: Path) -> int:
    return int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))


def display_frame(frame: numpy.ndarray):
    frame = cv2.resize(frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    cv2.imshow(TITLE, frame)
    cv2.waitKey(0)
    cv2.destroyWindow(TITLE)


if __name__ == '__main__':
    print(frame_num((next(get_videos()))))
    for frame in islice(get_frames(next(get_videos())), None, None, 50):
        display_frame(frame)
