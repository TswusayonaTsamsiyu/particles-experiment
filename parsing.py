import cv2
from pathlib import Path
from numpy import ndarray
from typing import Iterable


def parse_video(path: Path) -> cv2.VideoCapture:
    return cv2.VideoCapture(str(path))


def get_frames(path: Path) -> Iterable[ndarray]:
    video = parse_video(path)

    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame

    video.release()


def frame_num(path: Path) -> int:
    return int(parse_video(path).get(cv2.CAP_PROP_FRAME_COUNT))
