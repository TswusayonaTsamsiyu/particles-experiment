from typing import Iterable, Container, Callable, Sequence

from bettercv.types import Image
from bettercv.video import Frame
from bettercv.contours import Contour, draw_contours
from bettercv.display import Window, show, fit_to_screen, left_of

from .particle import ParticleEvent

ESC = 27
CLOSE_BTN = -1
EXIT_CODES = {ESC, CLOSE_BTN}


def exit_for(codes: Container[int]) -> Callable[[int], None]:
    return lambda key_code: exit() if key_code in codes else None


handle_key_code = exit_for(EXIT_CODES)


def display_particles(events: Iterable[ParticleEvent]) -> None:
    for event in events:
        handle_key_code(show([fit_to_screen(Window(
            draw_contours(event.best_snapshot.frame.image, [event.best_snapshot.contour]),
            str(event.best_snapshot)
        ))]))


def display_frame(frame: Frame, binary: Image, contours: Sequence[Contour]) -> None:
    right_window = Window(draw_contours(binary, contours),
                          title=f"Binary {frame} with contours")
    left_window = Window(frame.image,
                         title=f"Prepared {frame}",
                         position=left_of(right_window))
    handle_key_code(show(map(fit_to_screen, (right_window, left_window))))
