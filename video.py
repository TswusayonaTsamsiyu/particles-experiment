import cv2 as cv
from pathlib import Path
from typing import Generator
from dataclasses import dataclass

from utils import Image


@dataclass
class Frame:
    pixels: Image
    index: int


class Video:
    def __init__(self, path: Path):
        self.path = path
        self._cap = None

    def __enter__(self):
        self._cap = cv.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"Could not open video file at {self.path}")
        return self

    def __exit__(self, *exc_args):
        self._cap.release()
        self._cap = None
        return False

    def __len__(self):
        return self.frame_num

    def __iter__(self):
        return self.iter_frames()

    def _get_prop(self, prop: int) -> int:
        return int(self._cap.get(prop))

    def _next_frame_index(self) -> int:
        return self._get_prop(cv.CAP_PROP_POS_FRAMES)

    def _jump_to_frame(self, index: int) -> None:
        self._cap.set(cv.CAP_PROP_POS_FRAMES, index)

    def _read_next(self) -> Frame:
        success, frame = self._cap.read()
        if not success:
            raise IOError(f"Could not read frame at index {self._next_frame_index() - 1}")
        return Frame(frame, self._next_frame_index() - 1)

    @property
    def frame_num(self) -> int:
        return self._get_prop(cv.CAP_PROP_FRAME_COUNT)

    def read_frame_at(self, index: int) -> Frame:
        original_index = self._next_frame_index()
        self._jump_to_frame(index)
        frame = self._read_next()
        self._jump_to_frame(original_index)
        return frame

    def iter_frames(self, *,
                    start: int = 0,
                    stop: int = None,
                    jump: int = 1) -> Generator[Frame, None, None]:
        self._jump_to_frame(start)
        stop = self.frame_num if stop is None else min(stop, self.frame_num)
        while self._next_frame_index() < stop:
            frame = self._read_next()
            yield frame
            self._jump_to_frame(frame.index + jump)
