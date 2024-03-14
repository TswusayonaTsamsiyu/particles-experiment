import matplotlib.pyplot as plt

from pathlib import Path
from operator import attrgetter
from typing import Sequence, Callable

from cloudchamber.particle import Particle

from fs import load_particles, CSV_PATH

_HISTOGRAMS = (
    (lambda particle: particle.start.timestamp, "Track Appearance Timestamp [sec]"),
    (attrgetter("length"), "Track Length [px]"),
    (attrgetter("width"), "Track Width [px]"),
    (attrgetter("angle"), "Track Angle [deg]"),
    (attrgetter("curvature"), "Track Curvature [1/px]"),
    (attrgetter("intensity"), "Mean Track Intensity [0-255]"),
)


def _format_title(label: str) -> str:
    title = " ".join(label.replace("Track ", "").split()[:-1])
    return f"{title} Histogram"


def _format_filename(label: str) -> str:
    return _format_title(label).replace(" ", "_")


def _plot_hist(particles: Sequence[Particle], attr: Callable, label: str,
               save_path: str = None, show: bool = True) -> None:
    fig = plt.figure()
    plt.hist([attr(particle) for particle in particles])
    plt.xlabel(label)
    plt.ylabel("No. Particles")
    title = _format_title(label)
    plt.title(title)
    if save_path:
        print(f"Saving {title}")
        fig.savefig(save_path)
    if show:
        fig.show()


def plot_histograms(particles: Sequence[Particle],
                    save_dir: Path = None, show: bool = True) -> None:
    for hist in _HISTOGRAMS:
        _plot_hist(particles, *hist, show=show,
                   save_path=(save_dir / _format_filename(hist[1])).with_suffix(".svg") if save_dir else None)


if __name__ == '__main__':
    plot_histograms(load_particles(CSV_PATH / "20240109_122031.csv"))
