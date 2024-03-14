from time import time
from pathlib import Path
from argparse import ArgumentParser

from cloudchamber.detection import analyze_video
from cloudchamber.debugging import display_particles

from analysis import plot_histograms
from fs import save_particles, load_particles, CSV_PATH, GRAPH_PATH

START_TIME = 120
NUM_SECONDS = 60


def detect(path: Path) -> None:
    start = time()
    particles = analyze_video(path, START_TIME, START_TIME + NUM_SECONDS)
    print(f"Found {len(particles)} particles in {time() - start} seconds")
    save_particles(particles, CSV_PATH / path.with_suffix(".csv").name)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("action", choices=["detect", "display", "hist"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.action == "detect":
        detect(args.path)
    elif args.action == "display":
        display_particles(load_particles(args.path))
    elif args.action == "hist":
        plot_histograms(load_particles(args.path), save_dir=GRAPH_PATH, show=False)


if __name__ == '__main__':
    main()
