import bettercv.image as img
from bettercv.video import Frame

from .config import Config


def preprocess(frame: Frame, config: Config) -> Frame:
    return frame.with_image(
        img.grayscale(img.crop(img.scale(frame.image, config.scale_factor), *config.crop_box))
    )


def smooth(frame: Frame, config: Config) -> Frame:
    return frame.with_image(img.blur(frame.image, (config.blur_size, config.blur_size)))
