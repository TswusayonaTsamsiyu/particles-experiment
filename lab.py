from sys import argv
from time import time
from pathlib import Path
from argparse import ArgumentParser

from bettercv.video import Video

from cloudchamber.detection import analyze_video
from cloudchamber.debugging import display_particles

from fs import get_bg_videos, save_particles, read_particles, CSV_PATH

START_TIME = 120
NUM_SECONDS = 20


def detect(path: Path) -> None:
    start = time()
    particles = analyze_video(path, START_TIME, START_TIME + NUM_SECONDS)
    print(f"Finished in {time() - start} seconds")
    print(f"Found {len(particles)} particles")
    save_particles(particles, CSV_PATH / path.with_suffix(".csv").name)


def display(csv: Path) -> None:
    display_particles(read_particles(csv))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("action", choices=["detect", "display"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.action == "detect":
        detect(args.path)
    elif args.action == "display":
        display(args.path)


if __name__ == '__main__':
    main()
