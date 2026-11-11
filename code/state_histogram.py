"""Draw the automaton-state histogram for the saved vanilla model.

Each bar is one state from 0 to 255. Include begins at 128. The counts come
from the saved state file, not from a new fit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).resolve().parent / "results"
STATE = RESULTS / "vanilla_state.npz"
OUTPUT = RESULTS / "state_histogram.png"
INCLUDE = 128


def main() -> None:
    state = np.load(STATE)["state"]
    counts = np.bincount(state.ravel(), minlength=256)
    states = np.arange(256)
    colors = ["#1f8a8a" if state_id < INCLUDE else "#c4a035" for state_id in states]

    figure, axis = plt.subplots(figsize=(11.2, 4.6), facecolor="#2b2b2b")
    axis.set_facecolor("#2b2b2b")
    axis.bar(states, counts, width=1.0, color=colors, align="center")
    axis.set_yscale("log")
    axis.axvline(INCLUDE - 0.5, color="#e6c36a", linewidth=1.2)
    axis.set_xlim(-0.5, 255.5)
    axis.set_xlabel("Automaton state", color="#f4f4f4")
    axis.set_ylabel("Number of automata, log scale", color="#f4f4f4")
    axis.set_title(
        "Vanilla Tsetlin Machine automaton states",
        color="#e6c36a",
        fontsize=16,
        pad=12,
    )
    axis.tick_params(colors="#f4f4f4")
    for spine in axis.spines.values():
        spine.set_color("#f4f4f4")
    axis.text(
        0.98,
        0.95,
        (
            f"{int(counts[0]):,} at state 0\n"
            f"{int(counts[INCLUDE:].sum()):,} in Include, state {INCLUDE} or above\n"
            f"{int(counts.sum()):,} automata in total"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="#f4f4f4",
        fontsize=11,
    )
    figure.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=140, facecolor=figure.get_facecolor())
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
