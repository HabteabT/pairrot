"""Draw the 2D picture from the lecture: the Tsetlin Machine can abstain.

The points are a two-moon problem, not 20 Newsgroups pairs. Each coordinate
is turned into binary threshold features, because a Tsetlin Machine only sees
bits. Where no clause votes, the picture stays in the middle and is marked
"I do not know". A small neural network is trained on the same points and
fills that gap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier

from vanilla_tsetlin import VanillaTsetlinMachine


RESULTS = Path(__file__).resolve().parent / "results"
OUTPUT = RESULTS / "knows_when.png"
LEVELS = 28
GRID = 90


def thermometer(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (values[:, None] >= thresholds[None, :]).astype(np.uint8)


def encode(points: np.ndarray, x_levels: np.ndarray, y_levels: np.ndarray) -> np.ndarray:
    return np.hstack((thermometer(points[:, 0], x_levels), thermometer(points[:, 1], y_levels)))


def tm_confidence(machine: VanillaTsetlinMachine, features: np.ndarray) -> np.ndarray:
    """Map votes to 0..1. A 0-0 vote is 0.5: the machine does not know."""

    field = np.empty(len(features), dtype=np.float32)
    for index, row in enumerate(features):
        scores, _ = machine.predict_scores(row)
        if int(scores[0]) == 0 and int(scores[1]) == 0:
            field[index] = 0.5
        else:
            margin = (float(scores[1]) - float(scores[0])) / (2.0 * machine.T)
            field[index] = np.clip(0.5 + margin, 0.0, 1.0)
    return field


def nearest_distance(grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    distances = np.empty(len(grid), dtype=np.float64)
    for index, row in enumerate(grid):
        distances[index] = np.linalg.norm(points - row, axis=1).min()
    return distances


def pick_note(field: np.ndarray, mesh: np.ndarray, points: np.ndarray, unknown: bool) -> tuple[float, float]:
    """Choose an arrow target away from the points."""

    distance = nearest_distance(mesh, points)
    flat = field.ravel()
    away = distance > 0.35
    if unknown:
        wanted = away & (np.abs(flat - 0.5) < 0.08)
    else:
        wanted = away & (np.abs(flat - 0.5) > 0.35)
    if not wanted.any():
        wanted = away if away.any() else np.ones(len(mesh), dtype=bool)
    chosen = mesh[wanted][len(mesh[wanted]) // 3]
    return float(chosen[0]), float(chosen[1])


def main() -> None:
    points, labels = make_moons(n_samples=500, noise=0.14, random_state=457)
    pad = 0.45
    x_min, x_max = points[:, 0].min() - pad, points[:, 0].max() + pad
    y_min, y_max = points[:, 1].min() - pad, points[:, 1].max() + pad
    x_levels = np.linspace(x_min, x_max, LEVELS)
    y_levels = np.linspace(y_min, y_max, LEVELS)
    features = encode(points, x_levels, y_levels)

    machine = VanillaTsetlinMachine(
        n_features=features.shape[1],
        n_clauses=40,
        T=12,
        s=6.0,
        seed=457,
    )
    machine.fit(features, labels.astype(np.uint32), epochs=25)

    network = MLPClassifier(hidden_layer_sizes=(24, 24), max_iter=600, random_state=457)
    network.fit(points, labels)

    xs = np.linspace(x_min, x_max, GRID)
    ys = np.linspace(y_min, y_max, GRID)
    mesh = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    tm_field = tm_confidence(machine, encode(mesh, x_levels, y_levels)).reshape(GRID, GRID)
    nn_field = network.predict_proba(mesh)[:, 1].reshape(GRID, GRID)

    colors = LinearSegmentedColormap.from_list(
        "knows",
        ["#5b2d86", "#1f8a8a", "#f2e85c"],
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.4), facecolor="#2b2b2b")
    figure.suptitle(
        "A Tsetlin machine knows when it does not know",
        color="#e6c36a",
        fontsize=16,
        y=0.98,
    )
    panels = (
        (axes[0], tm_field, "Tsetlin machine", "I do not know", pick_note(tm_field, mesh, points, unknown=True)),
        (axes[1], nn_field, "Neural network", "I know", pick_note(nn_field, mesh, points, unknown=False)),
    )
    for axis, field, name, note, note_at in panels:
        image = axis.imshow(
            field,
            origin="lower",
            extent=(x_min, x_max, y_min, y_max),
            cmap=colors,
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        for label, color, edge in ((0, "#f4f4f4", "#222222"), (1, "#2a2a2a", "#f4f4f4")):
            chosen = points[labels == label]
            axis.scatter(
                chosen[:, 0],
                chosen[:, 1],
                s=18,
                c=color,
                edgecolors=edge,
                linewidths=0.3,
                label=f"y={label}",
            )
        axis.annotate(
            note,
            xy=note_at,
            xytext=(0.02, -0.16),
            textcoords=("data", "axes fraction"),
            color="#e6c36a",
            fontsize=12,
            arrowprops={"arrowstyle": "->", "color": "#e6c36a"},
        )
        axis.set_title(name, color="#e6c36a", fontsize=13, pad=16)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(facecolor="#f7f7f7", framealpha=0.9, loc="upper right")
        bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        bar.ax.tick_params(colors="#f4f4f4")

    figure.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=140, facecolor=figure.get_facecolor())
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
