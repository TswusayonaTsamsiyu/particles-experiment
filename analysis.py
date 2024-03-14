import matplotlib.pyplot as plt

from operator import attrgetter
from typing import Sequence, Callable

from cloudchamber.particle import Particle

from fs import load_particles, CSV_PATH

HISTOGRAMS = (
    (attrgetter("length"), "Track Length [px]"),
    (attrgetter("width"), "Track Width [px]"),
    (attrgetter("angle"), "Track Angle [deg]"),
    (attrgetter("curvature"), "Track Curvature [1/px]"),
    (attrgetter("intensity"), "Mean Track Intensity [0-255]"),
    (lambda particle: particle.start.timestamp, "Track Appearance Timestamp [sec]")
)


def format_title(label: str) -> str:
    title = " ".join(label.replace("Track ", "").split()[:-1])
    return f"{title} Histogram"


def plot_hist(particles: Sequence[Particle], attr: Callable, label: str) -> None:
    plt.hist([attr(particle) for particle in particles])
    plt.xlabel(label)
    plt.ylabel("No. Particles")
    plt.title(format_title(label))
    plt.show()


if __name__ == '__main__':
    particles = load_particles(CSV_PATH / "20240109_122031.csv")
    for hist in HISTOGRAMS:
        plot_hist(particles, *hist)
