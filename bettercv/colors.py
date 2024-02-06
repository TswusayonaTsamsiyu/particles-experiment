from random import randint
from typing import Tuple, Iterator, Sequence
from colorutils import random_rgb, Color as ColorBase

ColorTuple = Tuple[int, int, int]

NUM_HUES = 360
HUE_RANGE = (0, NUM_HUES - 1)


class Color(ColorBase, ColorTuple):
    def __getitem__(self, index: int) -> float:
        return self.bgr[index]

    def __iter__(self) -> Iterator[float]:
        return iter(self.bgr)

    def __repr__(self) -> str:
        return f"<Color({self.bgr})>"

    def __str__(self) -> str:
        return str(self.bgr)

    def __len__(self) -> int:
        return 3

    @property
    def bgr(self) -> ColorTuple:
        return _reversed_tuple(self.rgb)

    @bgr.setter
    def bgr(self, value: ColorTuple) -> None:
        self.rgb = _reversed_tuple(value)


def _reversed_tuple(t: Tuple) -> Tuple:
    return tuple(reversed(t))


def random_color() -> Color:
    return Color(random_rgb())


def random_hue() -> int:
    return randint(*HUE_RANGE)


def max_sv(hue: int) -> Color:
    return Color(hsv=(hue, 1, 1))


def max_spaced_hues(n: int, first: int = None) -> Sequence[int]:
    spacing = NUM_HUES // n
    first = first or random_hue()
    return [(first + spacing * i) % NUM_HUES for i in range(n)]
