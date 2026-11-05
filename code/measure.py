"""Score the saved vanilla Tsetlin Machine on the prepared feature files.

Run this file to print the same text the dashboard shows. The numbers are
predictions from the saved automata, not a stored accuracy sentence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vanilla_tsetlin import VanillaTsetlinMachine, literal_phrase


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "dataset" / "prepared" / "s457_f1000_t1000_v200_e200_val0p2"
RESULTS = Path(__file__).resolve().parent / "results"


def _accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float((targets == predictions).mean())


def _confusion(targets: np.ndarray, predictions: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((2, 2), dtype=int)
    for truth, prediction in zip(targets, predictions):
        matrix[int(truth), int(prediction)] += 1
    return matrix.tolist()


def _load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    stored = np.load(data_dir / f"features_{split}.npz")
    return stored["X"].astype(np.uint8), stored["y"].astype(np.uint32)


def _load_machine(results_dir: Path) -> VanillaTsetlinMachine:
    summary = json.loads((results_dir / "vanilla_pilot.json").read_text(encoding="utf-8"))
    hyper = summary["hyperparameters"]
    state = np.load(results_dir / "vanilla_state.npz")["state"]
    n_classes, n_clauses, n_literals = state.shape
    machine = VanillaTsetlinMachine(
        n_features=n_literals // 2,
        n_clauses=n_clauses,
        T=int(hyper["T"]),
        s=float(hyper["s"]),
        n_classes=n_classes,
        number_of_state_bits=int(hyper["number_of_state_bits"]),
        seed=int(hyper["seed"]),
    )
    machine.state = state
    return machine


def _clause_lines(machine: VanillaTsetlinMachine, vocabulary: list[str]) -> list[str]:
    lines = []
    for class_id in range(machine.n_classes):
        for clause_id in range(machine.n_clauses):
            included = machine.included_indices(class_id, clause_id)
            if len(included) == 0:
                continue
            phrases = [
                literal_phrase(int(index), machine.n_features, vocabulary) for index in included[:8]
            ]
            polarity = "positive" if clause_id % 2 == 0 else "negative"
            lines.append(
                f"class {class_id} clause {clause_id} ({polarity}): {', '.join(phrases)}"
            )
    return lines or ["No clause included a literal."]


def measure(data_dir: Path = DEFAULT_DATA, results_dir: Path = RESULTS) -> str:
    """Return the scoring printout. Raise FileNotFoundError if training has not been saved."""

    state_path = results_dir / "vanilla_state.npz"
    pilot_path = results_dir / "vanilla_pilot.json"
    if not state_path.exists() or not pilot_path.exists():
        raise FileNotFoundError(state_path)

    machine = _load_machine(results_dir)
    vocabulary = json.loads((data_dir / "vocabulary.json").read_text(encoding="utf-8"))["words"]
    lines = [
        "Scored the saved Tsetlin Machine on the feature files.",
        f"clauses {machine.n_clauses}, T {machine.T}, s {machine.s}",
        "",
    ]
    scored: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "val"):
        features, targets = _load_split(data_dir, split)
        predictions = machine.predict(features)
        scored[split] = (targets, predictions)
        lines.append(
            f"{split:5} X {tuple(features.shape)}  accuracy {_accuracy(targets, predictions):.3f}"
        )

    targets, predictions = scored["val"]
    matrix = _confusion(targets, predictions)
    lines.extend(
        [
            "",
            "validation confusion matrix, rows are true Y and columns are predicted Y",
            f"Y=0  predicted 0: {matrix[0][0]:4}   predicted 1: {matrix[0][1]:4}",
            f"Y=1  predicted 0: {matrix[1][0]:4}   predicted 1: {matrix[1][1]:4}",
            "",
            "included literals",
        ]
    )
    lines.extend(_clause_lines(machine, vocabulary))
    return "\n".join(lines)


if __name__ == "__main__":
    print(measure())
