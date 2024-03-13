from contextlib import ExitStack
from typing import Iterable, Container, Callable, Sequence

from bettercv.image import abc
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


def display_particles(particles: Iterable[Particle], **config) -> None:
    config = Config.merge(config)
    with ExitStack() as stack:
        videos = {path: stack.enter_context(Video(path))
                  for path in {particle.snapshot.ref.video for particle in particles}}
        for particle in particles:
            frame = preprocess(videos[particle.snapshot.ref.video][particle.snapshot.ref.index], config)
            handle_key_code(Window(
                draw_contours(abc(frame.image), [particle.snapshot.contour]),
                str(particle.snapshot)
            ).fit_to_screen().show())


def display_frame(frame: Frame, binary: Image, contours: Sequence[Contour]) -> None:
    right_window = Window(draw_contours(binary, contours),
                          title=f"Binary {frame} with contours").fit_to_screen()
    left_window = Window(frame.image,
                         title=f"Prepared {frame}",
                         position=left_of(right_window)).fit_to_screen()
    handle_key_code(show((right_window, left_window)))
