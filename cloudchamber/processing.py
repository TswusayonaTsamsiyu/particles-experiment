from typing import Generator, Iterable

import bettercv.image as img
from bettercv.video import Frame

from .config import Config


def preprocess(frame: Frame, config: Config) -> Frame:
    return frame.with_image(
        img.grayscale(img.crop(img.scale(frame.image, config.scale_factor), *config.crop_box))
    )


def smooth(frame: Frame, config: Config) -> Frame:
    return frame.with_image(img.blur(frame.image, (config.blur_size, config.blur_size)))


def has_tracks(threshold: float, min_thresh: float) -> bool:
    return threshold >= min_thresh


def binaries_with_tracks(frames: Iterable[Frame], config: Config) -> Generator[Frame, None, None]:
    for frame in frames:
        thresh, binary = img.threshold_otsu(frame.image)
        if has_tracks(thresh, config.min_threshold):
            yield frame.with_image(binary)
