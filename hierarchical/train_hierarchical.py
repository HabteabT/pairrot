"""Train the hierarchical Tsetlin Machine on the prepared pairs.

s = 1.0 and boost stays off, the same elimination rule as the vanilla run.
The test pairs are not read. Validation is scored after each epoch and the
best epoch is kept, which is the same rule as the vanilla trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hierarchical_tsetlin import HierarchicalTsetlinMachine, learns_a_pair_pattern


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "dataset" / "prepared" / "s457_f1000_t10000_v2000_e2000_val0p2"
RESULTS = Path(__file__).resolve().parent / "results"


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir / f"features_{split}.npz"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run dataset/prepare_data.py first.")
    stored = np.load(path)
    if set(stored.files) != {"X", "y"}:
        raise SystemExit(f"{path.name} should contain only X and y. Found {stored.files}.")
    return stored["X"].astype(np.uint8), stored["y"].astype(np.uint32)


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((2, 2), dtype=int)
    for truth, prediction in zip(targets, predictions):
        matrix[int(truth), int(prediction)] += 1
    return matrix.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hierarchical Tsetlin Machine.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--clauses", type=int, default=40)
    parser.add_argument("--leaves", type=int, default=4)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--s", type=float, default=1.0)
    parser.add_argument("--boost", type=int, choices=(0, 1), default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=457)
    parser.add_argument(
        "--continue-epochs",
        type=int,
        default=0,
        help="Train this many extra epochs from the saved tree instead of starting over.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs(args.s - 1.0) > 1e-9:
        raise SystemExit("This run keeps s = 1.0.")
    pair_check = learns_a_pair_pattern()
    print(f"Pair-pattern check: {pair_check:.1f}")
    if pair_check < 1.0:
        raise SystemExit("The hand-built tree did not vote for a shared word.")

    data_dir = args.data.resolve()
    x_train, y_train = load_split(data_dir, "train")
    x_val, y_val = load_split(data_dir, "val")
    vocabulary = json.loads((data_dir / "vocabulary.json").read_text(encoding="utf-8"))["words"]
    if x_train.shape[1] != len(vocabulary) * 2:
        raise SystemExit("The feature row is not two equal word sets.")
    print(
        f"Train X {tuple(x_train.shape)}, validation X {tuple(x_val.shape)}. "
        "Test pairs are not read."
    )
    machine = HierarchicalTsetlinMachine(
        n_words=len(vocabulary),
        n_clauses=args.clauses,
        n_leaves=args.leaves,
        T=args.T,
        s=args.s,
        seed=args.seed,
        boost_true_positive_feedback=args.boost,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state = machine.state.copy()
    first_epoch = 1
    last_epoch = args.epochs
    if args.continue_epochs > 0:
        saved_path = RESULTS / "hierarchical.json"
        state_path = RESULTS / "hierarchical_state.npz"
        if not saved_path.exists() or not state_path.exists():
            raise SystemExit("No saved hierarchical model to continue.")
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if Path(saved["data_dir"]).resolve() != data_dir:
            raise SystemExit("The saved hierarchical model was trained on a different dataset.")
        state = np.load(state_path)["state"]
        if tuple(state.shape) != tuple(machine.state.shape):
            raise SystemExit(
                f"Saved state shape {state.shape} does not match this run {machine.state.shape}."
            )
        machine.state = state
        progress_path = RESULTS / "hierarchical_progress.json"
        if progress_path.exists():
            stored = json.loads(progress_path.read_text(encoding="utf-8"))
            for row in stored.get("epoch_history") or []:
                history.append(
                    {
                        "epoch": int(row["epoch"]),
                        "train_accuracy": float(row.get("train_accuracy", row.get("train"))),
                        "validation_accuracy": float(row.get("validation_accuracy", row.get("validation"))),
                    }
                )
        best_epoch = int(saved.get("best_epoch") or 1)
        best_accuracy = float(saved["validation_accuracy"])
        best_state = machine.state.copy()
        last_finished = max((int(row["epoch"]) for row in history), default=best_epoch)
        first_epoch = last_finished + 1
        last_epoch = last_finished + args.continue_epochs
        args.epochs = last_epoch
        print(
            f"Continuing from the saved weights of epoch {best_epoch}, "
            f"validation accuracy {best_accuracy:.3f}. "
            f"Training epochs {first_epoch} to {last_epoch}."
        )

    def remember(epoch: int) -> float:
        nonlocal best_accuracy, best_epoch, best_state
        train_accuracy = float((machine.predict(x_train) == y_train).mean())
        val_predictions = machine.predict(x_val)
        val_accuracy = float((val_predictions == y_val).mean())
        history.append(
            {
                "epoch": epoch,
                "train_accuracy": train_accuracy,
                "validation_accuracy": val_accuracy,
            }
        )
        print(
            f"epoch {epoch}: train {train_accuracy:.3f}, validation {val_accuracy:.3f}",
            flush=True,
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_state = machine.state.copy()
            save(epoch, complete=False, predictions=val_predictions, train_accuracy=train_accuracy, val_accuracy=val_accuracy)
        progress = {
            "run_complete": False,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "validation": round(val_accuracy, 3),
            "best_validation": round(best_accuracy, 3),
            "validation_accuracy": best_accuracy,
            "epoch_history": [
                {"epoch": row["epoch"], "train": row["train_accuracy"], "validation": row["validation_accuracy"]}
                for row in history
            ],
        }
        (RESULTS / "hierarchical_progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
        return val_accuracy

    def save(epoch: int, complete: bool, predictions: np.ndarray, train_accuracy: float, val_accuracy: float) -> None:
        matrix = confusion_matrix(y_val, predictions)
        clause_text = machine.describe(vocabulary)
        summary = {
            "stage": "hierarchical",
            "implementation": (
                "Python Lecture 4 tree. Each clause is the AND of a Post A group and a Post B group. "
                "Each group adds its leaf conjunctions. s = 1.0. "
                + (
                    "Boost is on, so a firing leaf includes every true literal. "
                    if args.boost
                    else "Boost is off, so a true literal is not included. "
                )
                + "PyHierarchicalTsetlinMachineCUDA was not run; this computer has no NVIDIA GPU."
            ),
            "hierarchy": (
                f"{args.clauses} clauses per class, 2 groups, {args.leaves} leaves in each group"
            ),
            "data_dir": str(data_dir),
            "test_pairs_used": False,
            "train_shape": list(x_train.shape),
            "validation_shape": list(x_val.shape),
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "run_complete": complete,
            "best_epoch": epoch,
            "validation_confusion_matrix": {
                "rows": "true Y=0, true Y=1",
                "columns": "predicted Y=0, predicted Y=1",
                "values": matrix,
            },
            "hyperparameters": {
                "clauses_per_class": args.clauses,
                "leaves_per_group": args.leaves,
                "T": args.T,
                "s": args.s,
                "boost_true_positive_feedback": args.boost,
                "epochs": args.epochs,
                "seed": args.seed,
            },
        }
        (RESULTS / "hierarchical.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (RESULTS / "hierarchical_clauses.txt").write_text(clause_text, encoding="utf-8")
        np.savez_compressed(RESULTS / "hierarchical_state.npz", state=machine.state)

    for epoch in range(first_epoch, last_epoch + 1):
        machine.fit_epoch(x_train, y_train)
        remember(epoch)

    machine.state = best_state
    train_accuracy = float((machine.predict(x_train) == y_train).mean())
    val_predictions = machine.predict(x_val)
    val_accuracy = float((val_predictions == y_val).mean())
    save(best_epoch, complete=True, predictions=val_predictions, train_accuracy=train_accuracy, val_accuracy=val_accuracy)
    progress = {
        "run_complete": True,
        "epoch": last_epoch,
        "best_epoch": best_epoch,
        "validation": round(val_accuracy, 3),
        "best_validation": round(val_accuracy, 3),
        "validation_accuracy": val_accuracy,
        "epoch_history": [
            {"epoch": row["epoch"], "train": row["train_accuracy"], "validation": row["validation_accuracy"]}
            for row in history
        ],
    }
    (RESULTS / "hierarchical_progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    print(f"Best validation accuracy {val_accuracy:.3f} at epoch {best_epoch}")
    print(f"Wrote {RESULTS / 'hierarchical.json'}")


if __name__ == "__main__":
    main()
