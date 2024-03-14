from contextlib import ExitStack
from typing import Iterable, Container, Callable, Sequence

from bettercv.image import abc
from bettercv.track import Track
from bettercv.types import Image
from bettercv.video import Video, Frame
from bettercv.display import Window, show, left_of
from bettercv.contours import Contour, draw_contours

from .config import Config
from .particle import Particle
from .processing import preprocess

ESC = 27
CLOSE_BTN = -1
EXIT_CODES = {ESC, CLOSE_BTN}


def exit_for(codes: Container[int]) -> Callable[[int], None]:
    return lambda key_code: exit() if key_code in codes else None


handle_key_code = exit_for(EXIT_CODES)


def display_image(image: Image, title: str) -> None:
    handle_key_code(Window(image, title).fit_to_screen().show())


def display_particles(particles: Iterable[Particle], **config) -> None:
    config = Config.merge(config)
    with ExitStack() as stack:
        videos = {path: stack.enter_context(Video(path))
                  for path in {particle.snapshot.ref.video for particle in particles}}
        for particle in particles:
            frame = preprocess(videos[particle.snapshot.ref.video][particle.snapshot.ref.index], config)
            display_image(
                draw_contours(abc(frame.image), [particle.snapshot.contour]),
                str(particle.snapshot)
            )


def display_frame(frame: Frame) -> Frame:
    display_image(frame.image, str(frame))
    return frame


def display_contours(frame: Frame, contours: Sequence[Contour]) -> Frame:
    display_image(draw_contours(frame.image, contours), str(frame))
    return frame


def display_contours(frame: Frame, contours: Sequence[Contour]) -> Frame:
    display_frame(frame.with_image(draw_contours(frame.image, contours)))
    return frame


def display_track(track: Track, **config) -> Track:
    config = Config.merge(config)
    with Video(track[0].ref.video) as video:
        for snapshot in track:
            image = draw_contours(abc(preprocess(video[snapshot.ref.index], config).image), [snapshot.contour])
            handle_key_code(Window(image, title=f"#{snapshot.index}: {snapshot}").show())
    return track
