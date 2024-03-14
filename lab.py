from time import time
import argparse as ap
from pathlib import Path

from cloudchamber.detection import analyze_video
from cloudchamber.debugging import display_particles

from analysis import plot_histograms
from fs import save_particles, load_particles, CSV_PATH, GRAPH_PATH


def detect(path: Path, start: int, duration: int) -> None:
    start_time = time()
    particles = analyze_video(path, start, (start + duration) if duration else None)
    print(f"Found {len(particles)} particles in {time() - start_time} seconds")
    save_particles(particles, CSV_PATH / path.with_suffix(".csv").name)


def parse_args() -> ap.Namespace:
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(title="Available Actions", required=True, dest="action")
    # Detection options
    detect_parser = subparsers.add_parser("detect")
    detect_parser.add_argument("video", type=Path)
    detect_parser.add_argument("start", type=int, default=0)
    detect_parser.add_argument("duration", type=int, nargs="?")
    # Display options
    display_parser = subparsers.add_parser("display")
    display_parser.add_argument("csv", type=Path)
    # Histogram options
    hist_parser = subparsers.add_parser("hist")
    hist_parser.add_argument("csv", type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    match args.action:
        case "detect":
            detect(args.video, args.start, args.duration)
        case "display":
            display_particles(load_particles(args.csv))
        case "hist":
            plot_histograms(load_particles(args.csv), save_dir=GRAPH_PATH, show=False)


if __name__ == '__main__':
    main()
