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


def monochrome(frame: numpy.ndarray) -> numpy.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def threshold(frame: numpy.ndarray) -> numpy.ndarray:
    return 255 - cv2.threshold(frame, 20, 255, cv2.THRESH_BINARY)[1]


def blur(frame: numpy.ndarray) -> numpy.ndarray:
    return cv2.GaussianBlur(frame, (11, 11), 0)


def display_frame(frame: numpy.ndarray):
    frame = cv2.resize(frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    cv2.imshow(TITLE, frame)
    cv2.waitKey(0)
    cv2.destroyWindow(TITLE)


if __name__ == '__main__':
    # print(frame_num((list(get_videos())[1])))
    frames = islice(get_frames(list(get_videos())[1]), 150, None, 50)
    bg = threshold(monochrome(next(frames)))
    display_frame(bg)
    for frame in frames:
        display_frame(threshold(monochrome(frame)))
        display_frame(frame)
