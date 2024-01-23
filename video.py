import cv2
from pathlib import Path
from numpy import ndarray
from typing import Generator


class Video:
    def __init__(self, path: Path):
        self.path = path
        self._cap = None

    def __enter__(self):
        self._cap = cv2.VideoCapture(str(self.path))
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
        return self._get_prop(cv2.CAP_PROP_POS_FRAMES)

    def _jump_to_frame(self, index: int) -> None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    def _read_next(self) -> ndarray:
        success, frame = self._cap.read()
        if not success:
            raise IOError(f"Could not read frame at index {self._next_frame_index() - 1}")
        return frame

    @property
    def frame_num(self):
        return self._get_prop(cv2.CAP_PROP_FRAME_COUNT)

    def read_frame_at(self, index: int) -> ndarray:
        original_index = self._next_frame_index()
        self._jump_to_frame(index)
        frame = self._read_next()
        self._jump_to_frame(original_index)
        return frame

    def iter_frames(self, *,
                    start: int = 0,
                    stop: int = None,
                    jump: int = 1) -> Generator[ndarray, None, None]:
        self._jump_to_frame(start)
        stop = self.frame_num if stop is None else min(stop, self.frame_num)
        while self._next_frame_index() < stop:
            yield self._read_next()
            self._jump_to_frame(self._next_frame_index() + jump - 1)
