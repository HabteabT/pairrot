"""Draw training and validation accuracy for each saved epoch."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS = Path(__file__).resolve().parent / "results"
PILOT = RESULTS / "vanilla_pilot.json"
PROGRESS = RESULTS / "vanilla_progress.json"
OUTPUT = RESULTS / "accuracy_line.png"
HIERARCHICAL = Path(__file__).resolve().parents[1] / "hierarchical" / "results" / "hierarchical.json"


def _rows(payload: dict) -> list[dict]:
    raw = payload.get("history") or payload.get("epoch_history") or []
    rows = []
    for row in raw:
        train = row.get("train", row.get("train_accuracy"))
        validation = row.get("validation", row.get("validation_accuracy"))
        if train is None:
            continue
        item = {"epoch": row["epoch"], "train": float(train)}
        if validation is not None:
            item["validation"] = float(validation)
        rows.append(item)
    return rows


def load_hierarchical() -> tuple[float, float] | None:
    """One training score and one validation score. This model is not trained epoch by epoch."""

    if not HIERARCHICAL.exists():
        return None
    payload = json.loads(HIERARCHICAL.read_text(encoding="utf-8"))
    if not payload.get("run_complete", False):
        return None
    train = payload.get("train_accuracy")
    validation = payload.get("validation_accuracy")
    if train is None or validation is None:
        return None
    return float(train), float(validation)


def load_history() -> list[dict]:
    """Use the finished saved model. The progress file is only for an unfinished run."""

    if PILOT.exists():
        pilot = json.loads(PILOT.read_text(encoding="utf-8"))
        if pilot.get("run_complete", True):
            saved = _rows(pilot)
            if saved:
                return saved
    if PROGRESS.exists():
        live = _rows(json.loads(PROGRESS.read_text(encoding="utf-8")))
        if live:
            return live
    if not PILOT.exists():
        return []
    return _rows(json.loads(PILOT.read_text(encoding="utf-8")))


def main() -> None:
    history = load_history()
    if not history:
        raise SystemExit(f"{PILOT.name} has no epoch history. Train again, then redraw.")

    epochs = [row["epoch"] for row in history]
    train = [row["train"] for row in history]
    validation = [(row["epoch"], row["validation"]) for row in history if row.get("validation") is not None]
    hierarchical = load_hierarchical()

    figure, axis = plt.subplots(figsize=(11.2, 5.2), facecolor="#2b2b2b")
    axis.set_facecolor("#2b2b2b")
    axis.plot(epochs, train, color="#e6c36a", linewidth=2.2, marker="o", label="Vanilla training")
    if validation:
        axis.plot(
            [epoch for epoch, _ in validation],
            [score for _, score in validation],
            color="#7ec8c8",
            linewidth=2.2,
            marker="o",
            label="Vanilla validation",
        )
        best_epoch, best_score = max(validation, key=lambda item: item[1])
        axis.annotate(
            f"vanilla validation {best_score:.3f}",
            xy=(best_epoch, best_score),
            xytext=(-118, -22),
            textcoords="offset points",
            color="#7ec8c8",
            fontsize=11,
        )
    if hierarchical is not None:
        hier_train, hier_validation = hierarchical
        axis.axhline(
            hier_train,
            color="#f0d090",
            linewidth=1.6,
            linestyle=(0, (4, 3)),
            label=f"Hierarchical training {hier_train:.3f}",
        )
        axis.axhline(
            hier_validation,
            color="#b7f0c0",
            linewidth=1.8,
            linestyle=(0, (4, 3)),
            label=f"Hierarchical validation {hier_validation:.3f}",
        )
        axis.annotate(
            f"hierarchical validation {hier_validation:.3f}",
            xy=(epochs[-1], hier_validation),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            color="#b7f0c0",
            fontsize=11,
        )
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(epochs)
    axis.set_xlabel("Epoch", color="#f4f4f4")
    axis.set_ylabel("Accuracy", color="#f4f4f4")
    axis.set_title("Vanilla by epoch, hierarchical as level lines", color="#e6c36a", fontsize=16, pad=12)
    axis.tick_params(colors="#f4f4f4")
    for spine in axis.spines.values():
        spine.set_color("#f4f4f4")
    legend = axis.legend(
        loc="lower right",
        facecolor="#2b2b2b",
        edgecolor="#f4f4f4",
        labelcolor="#f4f4f4",
    )
    for text in legend.get_texts():
        text.set_color("#f4f4f4")
    figure.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=140, facecolor=figure.get_facecolor())
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
