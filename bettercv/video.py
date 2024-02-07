import cv2 as cv
from pathlib import Path
from datetime import timedelta
from dataclasses import dataclass
from typing import Generator, Union, List

from .types import Image


@dataclass
class Frame:
    image: Image
    index: int
    timestamp: timedelta

    def with_image(self, image: Image) -> "Frame":
        return Frame(image, self.index, self.timestamp)

    def __str__(self) -> str:
        return repr(self).strip("<>")

    def __repr__(self) -> str:
        return f"<Frame {self.index} at {self.timestamp}>"


class Video:
    def __init__(self, path: Union[Path, str]) -> None:
        self.path = Path(path)
        self._cap = None

    def __enter__(self) -> "Video":
        self._cap = cv.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"Could not open video file at {self.path}")
        return self

    def __exit__(self, *exc_args) -> bool:
        self._cap.release()
        self._cap = None
        return False

    def __len__(self) -> int:
        return self.frame_num

    def __iter__(self) -> Generator[Frame, None, None]:
        return self.iter_frames()

    def __getitem__(self, index: Union[int, slice]) -> Union[Frame, List[Frame]]:
        if isinstance(index, slice):
            return list(self.iter_frames(start=index.start or 0,
                                         stop=index.stop,
                                         jump=index.step or 1))
        return self.read_frame_at(index)

    def __repr__(self) -> str:
        return f"<Video {self.name}>"

    def _get_prop(self, prop: int) -> int:
        return int(self._cap.get(prop))

    def _current_timestamp(self) -> timedelta:
        return timedelta(milliseconds=self._get_prop(cv.CAP_PROP_POS_MSEC))

    def _next_frame_index(self) -> int:
        return self._get_prop(cv.CAP_PROP_POS_FRAMES)

    def _jump_to_frame(self, index: int) -> None:
        self._cap.set(cv.CAP_PROP_POS_FRAMES, index)

    def _read_next(self) -> Frame:
        success, frame = self._cap.read()
        if not success:
            raise IOError(f"Could not read frame at index {self._next_frame_index() - 1}")
        return Frame(frame, self._next_frame_index() - 1, self._current_timestamp())

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def frame_num(self) -> int:
        return self._get_prop(cv.CAP_PROP_FRAME_COUNT)

    @property
    def width(self):
        return self._get_prop(cv.CAP_PROP_FRAME_WIDTH)

    @property
    def height(self):
        return self._get_prop(cv.CAP_PROP_FRAME_HEIGHT)

    @property
    def fps(self):
        return self._get_prop(cv.CAP_PROP_FPS)

    @property
    def duration(self) -> timedelta:
        return timedelta(seconds=self.frame_num / self.fps)

    def index_at(self, time: Union[int, timedelta]) -> int:
        return (time if isinstance(time, int) else time.total_seconds()) * self.fps

    def timestamp_at(self, index: int) -> timedelta:
        return timedelta(seconds=index / self.fps)

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
        if jump == 0:
            raise ValueError("Jump cannot be zero!")
        self._jump_to_frame(start)
        stop = self.frame_num if stop is None else min(stop, self.frame_num)
        while self._next_frame_index() < stop:
            frame = self._read_next()
            yield frame
            if jump > 1:
                self._jump_to_frame(frame.index + jump)
