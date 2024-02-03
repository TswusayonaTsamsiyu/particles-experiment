from typing import Container, Callable

from .types import Position

MAX_PIXEL_VALUE = 255
MIN_PIXEL_VALUE = 0


def exit_for(codes: Container[int]) -> Callable[[int], None]:
    return lambda key_code: exit() if key_code in codes else None


def distance(point1: Position, point2: Position) -> float:
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** .5
