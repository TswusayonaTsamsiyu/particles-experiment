import cv2 as cv
from pathlib import Path
from datetime import timedelta
from dataclasses import dataclass
from typing import Generator, Union, List

from .types import Image


@dataclass
class Ref:
    """
    A reference to a location in a video.

    video (Path): The path to the video file from which the frame is taken
    index (int): The index of the frame in the video file
    timestamp (int): The timestamp of the frame in the video file, in seconds
    """
    video: Path
    index: int
    timestamp: float

    @property
    def time(self) -> timedelta:
        """
        The delta between the beginning of the video and the time when the frame occurs
        """
        return timedelta(seconds=self.timestamp)


@dataclass
class Frame:
    """
    A frame in a video.

    Attributes:
        image (Image): The actual image in the frame
        ref (Ref): A reference of where the frame is taken from
    """
    image: Image
    ref: Ref

    def with_image(self, image: Image) -> "Frame":
        """
        Modify the frame with a new image.
        Useful to preserve frame attributes while applying image analysis.

        Args:
            image: The new image to substitute for the current image

        Returns:
            A new frame object with the replaced image
        """
        return Frame(image, self.ref)

    def __str__(self) -> str:
        return repr(self).strip("<>")

    def __repr__(self) -> str:
        return f"<Frame {self.ref.index}, {self.ref.time} from {self.ref.video.name}>"


class Video:
    """
    A video that is read from a file.
    Can be used as a context manager.
    Acts as a sequence of `Frame`s (supports `len`, indexing, slicing and iteration).

    Example:
        ```
        with Video("/path/to/video.mp4") as video:
            print(video[0])  # prints first frame
            print(video.fps)
            for frame in video:
                # Do some image analysis
        ```

    Args:
        path (str or Path): The path to the video file

    Attributes:
        path (Path): The path to the video file
    """

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
        """
        Returns:
            Whether the video is open for reading, or not
        """
        return bool(self._cap)

    def _raise_if_closed(self) -> None:
        """
        Raises:
            OSError: if the video is closed for reading
        """
        if not self._is_open():
            raise OSError(f"{self} is closed.")

    def _get_prop(self, prop: int) -> int:
        """
        Reads a property of the video.
        Supported valued are the different `cv.CAP_PROP_*`.

        Args:
            prop: The name of the property to read

        Returns:
            The value of the property
        """
        self._raise_if_closed()
        return int(self._cap.get(prop))

    def _current_timestamp(self) -> float:
        """
        Returns:
            The current position of the video capture pointer in terms of time from the start of the video
        """
        return self._get_prop(cv.CAP_PROP_POS_MSEC) / 1000

    def _next_frame_index(self) -> int:
        """
        Returns:
            The index of the next frame to be read from the video
        """
        return self._get_prop(cv.CAP_PROP_POS_FRAMES)

    def _jump_to_frame(self, index: int) -> None:
        """
        Changes the position of the video capture pointer to a specific index.

        Args:
            index: the index of the frame to be read next
        """
        self._cap.set(cv.CAP_PROP_POS_FRAMES, index)

    def _read_next(self) -> Frame:
        """
        Reads the next frame from the video file, based on the current position of the video capture pointer.

        Returns:
            The next frame from the video

        Raises:
            OSError: if the frame could not be read
        """
        success, frame = self._cap.read()
        if not success:
            raise OSError(f"Could not read frame at index {self._next_frame_index() - 1}")
        return Frame(frame, Ref(self.path, self._next_frame_index() - 1, self._current_timestamp()))

    def open(self) -> "Video":
        """
        Opens the video for reading.
        Don't forget to close the video when finishing.
        It is recommended to use the context manager syntax instead (e.g. `with Video("...") as video: ...`).

        Returns:
            The opened video object

        Raises:
            OSError: if the video file failed to open
        """
        if not self._is_open():
            self._cap = cv.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise OSError(f"Could not open video file at {self.path}")
        return self

    def close(self) -> None:
        """
        Closes the video for reading.
        """
        if self._is_open():
            self._cap.release()
            self._cap = None

    @property
    def name(self) -> str:
        """
        The name of the video file
        """
        return self.path.name

    @property
    def frame_num(self) -> int:
        """
        The number of frames in the video
        """
        return self._get_prop(cv.CAP_PROP_FRAME_COUNT)

    @property
    def width(self) -> int:
        """
        The width of the frames in the video, in pixels
        """
        return self._get_prop(cv.CAP_PROP_FRAME_WIDTH)

    @property
    def height(self) -> int:
        """
        The height of the frames in the video, in pixels
        """
        return self._get_prop(cv.CAP_PROP_FRAME_HEIGHT)

    @property
    def fps(self) -> int:
        """
        The frame rate of the video, in frames per second
        """
        return self._get_prop(cv.CAP_PROP_FPS)

    @property
    def duration(self) -> timedelta:
        """
        The duration (temporal length) of the video
        """
        return timedelta(seconds=self.frame_num / self.fps)

    def index_at(self, time: Union[int, timedelta]) -> int:
        """
        Converts a timestamp to the index of the frame at that timestamp.

        Args:
            time: The time delta from the start of the video of the relevant frame

        Returns:
            The appropriate index of the frame occurring at the given time
        """
        return (time if isinstance(time, int) else time.total_seconds()) * self.fps

    def timestamp_at(self, index: int) -> timedelta:
        """
        Converts an index of a frame in the video to the timestamp at which it occurs.

        Args:
            index: The index of frame in the video

        Returns:
            The appropriate time delta from the start of the video when it reaches the given index
        """
        return timedelta(seconds=index / self.fps)

    def read_frame_at(self, index: int) -> Frame:
        """
        Reads a frame at a given index in the video.

        Note that this method does not change the position of the underlying video capture pointer,
        so it is safe to use during iteration over the video.

        Args:
            index: The index of the frame to read

        Returns:
            The read frame

        Raises:
            OSError: if the video is not open for reading, or if the frame could not be read
            IndexError: if the index is out of bounds from the length of the video
        """
        self._raise_if_closed()
        if index < 0 or index >= self.frame_num:
            raise IndexError("No frame available at the given index! Check the length of the video.")
        original_index = self._next_frame_index()
        self._jump_to_frame(index)
        frame = self._read_next()
        self._jump_to_frame(original_index)
        return frame

    def iter_frames(self, *,
                    start: int = 0,
                    stop: int = None,
                    jump: int = 1) -> Generator[Frame, None, None]:
        """
        Yields frames from a specified slice of the video.

        Note that this is a generator function, which will lazy-load each frame when needed,
        so it will not fill your memory (if used wisely).

        Please also note that due to underlying IO optimizations, it is recommended to use this
        method (or slice the video object directly), instead of applying a slice to the iterator
        returned by this function (or by iterating over the video object directly). This method
        makes sure frames are not read within the jump intervals.

        Args:
            start: The index of the first frame to read
            stop: The index of the last frame to read
            jump: How many indices to jump between frame reads. Must be positive.

        Returns:
            A generator of frames

        Raises:
            OSError: if the video is not open for reading
            ValueError: if the jump is not positive
        """
        if jump == 0:
            raise ValueError("Jump cannot be zero!")
        if jump < 0:
            raise ValueError("Video does not support backwards reading. You may want to use `reversed` instead.")
        self._raise_if_closed()
        self._jump_to_frame(start or 0)
        stop = self.frame_num if stop is None else min(stop, self.frame_num)
        while self._next_frame_index() < stop:
            frame = self._read_next()
            yield frame
            # This condition is an optimization to avoid the additional IO operation when jump = 1,
            # which is redundant since `self._read_next` automatically advances the capture pointer.
            if jump > 1:
                self._jump_to_frame(frame.ref.index + jump)
