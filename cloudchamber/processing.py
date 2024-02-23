from more_itertools import chunked
from typing import Generator, Iterable

import bettercv.image as img
from bettercv.video import Frame
from bettercv.display import Window

from .config import Config


def preprocess(frame: Frame, config: Config) -> Frame:
    return frame.with_image(img.grayscale(img.scale(frame.image, config.scale_factor)))


def smooth(frame: Frame, config: Config) -> Frame:
    return frame.with_image(img.blur(frame.image, (config.blur_size, config.blur_size)))


def subtract_bg(frames: Iterable[Frame], config: Config) -> Generator[Frame, None, None]:
    for batch in chunked(frames, config.bg_batch_size):
        if config.prints:
            print(f"Computing BG for {batch[0].ref.index}-{batch[-1].ref.index}")
        bg = img.avg(frame.image for frame in batch[::config.bg_jump])
        if config.display:
            Window(bg, "Avg BG").fit_to_screen().show()
        for frame in batch:
            yield frame.with_image(img.subtract(frame.image, bg))


def subtract_bg_2(frames: Iterable[Frame], config: Config) -> Generator[Frame, None, None]:
    had_tracks = False
    bg = next(iter(frames)).image
    for frame in frames:
        thresh, binary = img.threshold_otsu(img.subtract(frame.image, bg))
        if has_tracks(thresh, config.min_threshold):
            had_tracks = True
            yield frame.with_image(binary)
        else:
            if had_tracks:
                if config.prints:
                    print(f"New BG is {frame}")
                bg = frame.image
            had_tracks = False


def has_tracks(threshold: float, min_thresh: float) -> bool:
    return threshold > min_thresh


def binaries_with_tracks(frames: Iterable[Frame], config: Config) -> Generator[Frame, None, None]:
    for frame in frames:
        thresh, binary = img.threshold_otsu(frame.image)
        if has_tracks(thresh, config.min_threshold):
            yield frame.with_image(binary)
