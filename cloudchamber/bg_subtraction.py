from itertools import tee
from more_itertools import chunked
from typing import Iterable, Generator

import bettercv.image as img
from bettercv.video import Frame
from bettercv.display import Window

from .config import Config


def has_tracks(threshold: float, min_thresh: float) -> bool:
    return threshold >= min_thresh


def has_tracks_2(frame: Frame) -> bool:
    return img.mean(frame.image)[0] > 0.5


def print_mean(frame: Frame) -> Frame:
    mean = img.mean(frame.image)
    print(f"{frame.ref.index}: {mean}")
    return frame


def binaries_with_tracks(frames: Iterable[Frame], config: Config) -> Generator[Frame, None, None]:
    for frame in frames:
        thresh, binary = img.threshold_otsu(frame.image)
        if has_tracks(thresh, config.min_threshold):
            yield frame.with_image(binary)


def binaries_with_tracks_2(frames: Iterable[Frame], config: Config) -> Generator[Frame, None, None]:
    for frame in frames:
        if has_tracks_2(frame):
            yield frame.with_image(img.threshold_binary(frame.image, 1))


def subtract_bg_avg(frames: Iterable[Frame], config: Config) -> Generator[Frame, None, None]:
    for batch in chunked(frames, config.bg_batch_size):
        if config.prints:
            print(f"Computing BG for {batch[0].ref.index}-{batch[-1].ref.index}")
        bg = img.avg(frame.image for frame in batch[::config.bg_jump])
        if config.display:
            Window(bg, "Avg BG").fit_to_screen().show()
        for frame in batch:
            yield frame.with_image(img.subtract(frame.image, bg))


def subtract_bg_replace(frames: Iterable[Frame], config: Config) -> Generator[Frame, None, None]:
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


def subtract_bg_mog2(frames: Iterable[Frame]) -> Generator[Frame, None, None]:
    frames, fg_masks = tee(frames)
    fg_masks = img.subtract_bg((frame.image for frame in fg_masks), detectShadows=False)
    return (frame.with_image(fg) for frame, fg in zip(frames, fg_masks))


def subtract_bg(frames: Iterable[Frame], config: Config) -> Generator[Frame, None, None]:
    match config.bg_method:
        case "mog2":
            return subtract_bg_mog2(frames)
        case "avg":
            return binaries_with_tracks(subtract_bg_avg(frames, config), config)
        case "replace":
            return subtract_bg_replace(frames, config)
