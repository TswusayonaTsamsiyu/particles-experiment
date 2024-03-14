from more_itertools import chunked
from typing import Iterable, Generator

import bettercv.image as img
from bettercv.video import Frame
from bettercv.display import Window

from .config import Config
from .processing import has_tracks, binaries_with_tracks


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
    fg = img.subtract_bg((frame.image for frame in frames), detectShadows=False)
    return (frame.with_image(next(fg)) for frame in frames)


def subtract_bg(frames: Iterable[Frame], config: Config) -> Generator[Frame, None, None]:
    match config.bg_method:
        case "mog2":
            return subtract_bg_mog2(frames)
        case "avg":
            return binaries_with_tracks(subtract_bg_avg(frames, config), config)
        case "replace":
            return subtract_bg_replace(frames, config)