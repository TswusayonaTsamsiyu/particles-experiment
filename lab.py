from sys import argv
from time import time
from pathlib import Path

from bettercv.video import Video

from cloudchamber.detection import detect_tracks
from cloudchamber.debugging import display_particles

from fs import get_bg_videos, save_particles, CSV_PATH

START_TIME = 120
NUM_SECONDS = 20


def analyze_video(path: Path) -> None:
    start = time()
    with Video(path) as video:
        print(f"Parsing {video.name}...")
        print(f"Video has {video.frame_num} frames.")
        particles = detect_tracks(video.iter_frames(
            start=video.index_at(START_TIME),
            stop=video.index_at(START_TIME + NUM_SECONDS)
        ))
        print(f"Num tracks found: {len(particles)}")
        print(f"Num particle events found: {len(particles)}")
        print(f"Finished in {time() - start} seconds.")
        display_particles(particles)
        save_particles(particles, CSV_PATH / path.with_suffix(".csv").name)


if __name__ == '__main__':
    video_path = argv[1] if len(argv) > 1 else get_bg_videos()[1]
    analyze_video(video_path)
