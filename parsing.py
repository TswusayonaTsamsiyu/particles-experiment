import cv2
from pathlib import Path
from numpy import ndarray
from typing import Iterable
from contextlib import contextmanager


@contextmanager
def parse_video(path: Path) -> cv2.VideoCapture:
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise IOError(f"Could not open video file at {path}")
    yield video
    video.release()


def iter_frames(video: cv2.VideoCapture) -> Iterable[ndarray]:
    while True:
        success, frame = video.read()
        if not success:
            break
        yield frame


def frame_num(video: cv2.VideoCapture) -> int:
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))
