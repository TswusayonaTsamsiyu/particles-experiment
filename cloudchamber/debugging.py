from typing import Iterable, Container, Callable, Sequence

from bettercv.types import Image
from bettercv.video import Frame
from bettercv.contours import Contour, draw_contours
from bettercv.display import Window, show, left_of

from .particle import ParticleTrack

ESC = 27
CLOSE_BTN = -1
EXIT_CODES = {ESC, CLOSE_BTN}


def exit_for(codes: Container[int]) -> Callable[[int], None]:
    return lambda key_code: exit() if key_code in codes else None


handle_key_code = exit_for(EXIT_CODES)


def display_particles(events: Iterable[ParticleTrack]) -> None:
    for event in events:
        handle_key_code(Window(
            draw_contours(event.snapshot.frame.image, [event.snapshot.contour]),
            str(event.snapshot)
        ).fit_to_screen().show())


def display_frame(frame: Frame, binary: Image, contours: Sequence[Contour]) -> None:
    right_window = Window(draw_contours(binary, contours),
                          title=f"Binary {frame} with contours").fit_to_screen()
    left_window = Window(frame.image,
                         title=f"Prepared {frame}",
                         position=left_of(right_window)).fit_to_screen()
    handle_key_code(show((right_window, left_window)))
