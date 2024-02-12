from sys import argv
from time import time
from pathlib import Path

from cloudchamber.detection import analyze_video
from cloudchamber.debugging import display_particles

from fs import get_bg_videos, save_particles, CSV_PATH

START_TIME = 120
NUM_SECONDS = 20


def main(path: Path) -> None:
    start = time()
    particles = analyze_video(path, START_TIME, START_TIME + NUM_SECONDS)
    print(f"Finished in {time() - start} seconds")
    print(f"Found {len(particles)} particles")
    display_particles(particles)
    save_particles(particles, CSV_PATH / path.with_suffix(".csv").name)


if __name__ == '__main__':
    video_path = argv[1] if len(argv) > 1 else get_bg_videos()[1]
    main(video_path)
