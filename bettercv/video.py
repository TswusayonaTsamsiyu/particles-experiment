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
        return self.open()

    def __exit__(self, *exc_args) -> bool:
        self.close()
        return False

    def __len__(self) -> int:
        return self.frame_num

    def __iter__(self) -> Generator[Frame, None, None]:
        return self.iter_frames()

    def __getitem__(self, index: Union[int, slice]) -> Union[Frame, List[Frame]]:
        if isinstance(index, slice):
            return list(self.iter_frames(start=index.start or 0,
                                         stop=index.stop,
                                         jump=index.step if index.step is not None else 1))
        return self.read_frame_at(index)

    def __str__(self):
        return repr(self).strip("<>")

    def __repr__(self) -> str:
        return f"<Video {self.name}>"

    def _is_open(self) -> bool:
        return bool(self._cap)

    def _raise_if_closed(self) -> None:
        if not self._is_open():
            raise RuntimeError(f"{self} is closed.")

    def _get_prop(self, prop: int) -> int:
        self._raise_if_closed()
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

    def open(self) -> "Video":
        if not self._is_open():
            self._cap = cv.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise IOError(f"Could not open video file at {self.path}")
        return self

    def close(self) -> None:
        if self._is_open():
            self._cap.release()
            self._cap = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def frame_num(self) -> int:
        return self._get_prop(cv.CAP_PROP_FRAME_COUNT)

    @property
    def width(self) -> int:
        return self._get_prop(cv.CAP_PROP_FRAME_WIDTH)

    @property
    def height(self) -> int:
        return self._get_prop(cv.CAP_PROP_FRAME_HEIGHT)

    @property
    def fps(self) -> int:
        return self._get_prop(cv.CAP_PROP_FPS)

    @property
    def duration(self) -> timedelta:
        return timedelta(seconds=self.frame_num / self.fps)

    def index_at(self, time: Union[int, timedelta]) -> int:
        return (time if isinstance(time, int) else time.total_seconds()) * self.fps

    def timestamp_at(self, index: int) -> timedelta:
        return timedelta(seconds=index / self.fps)

    def read_frame_at(self, index: int) -> Frame:
        self._raise_if_closed()
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
        if jump < 0:
            raise ValueError("Video does not support backwards reading. You may want to use `reversed` instead.")
        self._raise_if_closed()
        self._jump_to_frame(start)
        stop = self.frame_num if stop is None else min(stop, self.frame_num)
        while self._next_frame_index() < stop:
            frame = self._read_next()
            yield frame
            if jump > 1:
                self._jump_to_frame(frame.index + jump)
