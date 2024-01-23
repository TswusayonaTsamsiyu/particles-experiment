import cv2
from pathlib import Path
from numpy import ndarray
from typing import Generator
from contextlib import contextmanager


@contextmanager
def parse_video(path: Path) -> cv2.VideoCapture:
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise IOError(f"Could not open video file at {path}")
    yield video
    video.release()


def iter_frames(video: cv2.VideoCapture,
                start: int = 0, stop: int = None, jump: int = 1) -> Generator[ndarray, None, None]:
    jump_to_frame(video, start)
    index = 0
    while index < (stop or frame_num(video)):
        success, frame = video.read()
        if not success:
            break
        yield frame
        index = next_frame_index(video)
        jump_to_frame(video, index + jump - 1)


def frame_num(video: cv2.VideoCapture) -> int:
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))


def next_frame_index(video: cv2.VideoCapture) -> int:
    return int(video.get(cv2.CAP_PROP_POS_FRAMES))


def jump_to_frame(video: cv2.VideoCapture, index: int) -> None:
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
