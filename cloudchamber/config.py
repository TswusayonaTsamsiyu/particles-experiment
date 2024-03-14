from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class Config:
    # Preprocessing
    blur_size: int = 15
    scale_factor: float = 0.6
    crop_box: Tuple[int, int, int, int] = (50, 50, 0, 0)
    # Tracking
    track_distance: int = 30
    # Joining
    dist_close: int = 30
    # Filtering
    min_contour_size: int = 500
    min_track_length: int = 10
    min_threshold: int = 3
    # BG computation
    bg_method: str = "avg"  # or "replace"
    bg_jump: int = 5
    bg_batch_size: int = 200
    # Debug
    prints: bool = True
    display: bool = False

    @classmethod
    def merge(cls, config: "Dict") -> "Config":
        return cls(**config)
