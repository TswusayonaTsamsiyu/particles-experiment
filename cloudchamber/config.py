from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class Config:
    # Preprocessing
    blur_size: int = 15
    scale_factor: float = 0.6
    crop_box: Tuple[int, int, int, int] = (35, 20, 0, 0)
    # BG computation
    bg_method: str = "mog2"  # "mog2"/"avg"/"replace"
    bg_jump: int = 5
    bg_batch_size: int = 200
    # Thresholding
    min_threshold: int = 1
    # Contour Filtering
    min_contour_size: int = 500
    min_aspect_ratio: float = 3
    max_contour_width: int = 100
    # Contour Joining
    dist_close: int = 30
    # Contour Tracking
    track_distance: int = 30
    # Track Filtering
    min_track_length: int = 10
    # Debugging
    prints: bool = True
    display: bool = False

    @classmethod
    def merge(cls, config: "Dict") -> "Config":
        return cls(**config)
