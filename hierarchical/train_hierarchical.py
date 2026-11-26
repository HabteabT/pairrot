"""Train a second Tsetlin Machine on top of the saved vanilla clauses.

The pair task and s = 1.0 stay. The first level is the vanilla machine, which
already reads both posts. The second level is another Tsetlin Machine. Its
features are the vanilla clause outputs plus one bit per vocabulary word for
"this word is in both posts." Boost is on for this level so a firing clause
can include a true literal. The test pairs are not read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from hierarchical_tsetlin import learns_a_pair_pattern


REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "code"
DEFAULT_DATA = REPO / "dataset" / "prepared" / "s457_f300_t30000_v2000_e2000_val0p2_fulltext_topical"
RESULTS = Path(__file__).resolve().parent / "results"
VANILLA_STATE = CODE / "results" / "vanilla_state.npz"
VANILLA_PILOT = CODE / "results" / "vanilla_pilot.json"


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


def load_vanilla(n_features: int):
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    from vanilla_tsetlin import VanillaTsetlinMachine

    if not VANILLA_STATE.exists() or not VANILLA_PILOT.exists():
        raise SystemExit("The vanilla model is missing. Train it before the hierarchical level.")
    pilot = json.loads(VANILLA_PILOT.read_text(encoding="utf-8"))
    hyper = pilot["hyperparameters"]
    machine = VanillaTsetlinMachine(
        n_features=n_features,
        n_clauses=int(hyper["clauses_per_class"]),
        T=int(hyper["T"]),
        s=float(hyper["s"]),
        boost_true_positive_feedback=int(hyper["boost_true_positive_feedback"]),
        seed=int(hyper["seed"]),
    )
    machine.state = np.load(VANILLA_STATE)["state"]
    return machine, float(pilot["validation_accuracy"])


def clause_and_shared_features(machine, features: np.ndarray) -> np.ndarray:
    """Vanilla clause outputs, then one bit for each word present in both posts."""

    n_words = features.shape[1] // 2
    included = [(machine.state[class_id] >= machine.include_threshold) for class_id in range(2)]
    bits = np.zeros((len(features), machine.n_clauses * 2), dtype=np.uint8)
    for start in range(0, len(features), 400):
        block = features[start : start + 400]
        literal_false = np.concatenate((1 - block, block), axis=1).astype(np.float32)
        columns = []
        for mask in included:
            violations = literal_false @ mask.T.astype(np.float32)
            fired = (violations == 0) & mask.any(axis=1)[None, :]
            columns.append(fired.astype(np.uint8))
        bits[start : start + len(block)] = np.concatenate(columns, axis=1)
    shared = (features[:, :n_words] & features[:, n_words:]).astype(np.uint8)
    return np.concatenate((bits, shared), axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the second Tsetlin Machine level.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--clauses", type=int, default=200)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--s", type=float, default=1.0)
    parser.add_argument("--boost", type=int, choices=(0, 1), default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=457)
    parser.add_argument(
        "--continue-epochs",
        type=int,
        default=0,
        help="Train this many more epochs from the saved second-level weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs(args.s - 1.0) > 1e-9:
        raise SystemExit("This run keeps s = 1.0.")
    if learns_a_pair_pattern() < 1.0:
        raise SystemExit("The hand-built tree did not vote for a shared word.")
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    from vanilla_tsetlin import VanillaTsetlinMachine

    data_dir = args.data.resolve()
    x_train, y_train = load_split(data_dir, "train")
    x_val, y_val = load_split(data_dir, "val")
    vanilla, vanilla_val = load_vanilla(x_train.shape[1])
    print(
        f"Train X {tuple(x_train.shape)}, validation X {tuple(x_val.shape)}. "
        f"Vanilla validation {vanilla_val:.3f}. Test pairs are not read.",
        flush=True,
    )
    print("Building clause outputs and shared-word bits.", flush=True)
    upper_train = clause_and_shared_features(vanilla, x_train)
    upper_val = clause_and_shared_features(vanilla, x_val)
    print(f"Second level features {tuple(upper_train.shape)}.", flush=True)
    machine = VanillaTsetlinMachine(
        n_features=upper_train.shape[1],
        n_clauses=args.clauses,
        T=args.T,
        s=args.s,
        boost_true_positive_feedback=args.boost,
        seed=args.seed,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    history = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state = machine.state.copy()
    best_predictions = None
    best_train = 0.0
    first_epoch = 1
    last_epoch = args.epochs
    if args.continue_epochs > 0:
        saved_path = RESULTS / "hierarchical.json"
        state_path = RESULTS / "hierarchical_state.npz"
        if not saved_path.exists() or not state_path.exists():
            raise SystemExit("No saved second-level model to continue.")
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if Path(saved["data_dir"]).resolve() != data_dir:
            raise SystemExit("The saved second-level model was trained on a different dataset.")
        state = np.load(state_path)["state"]
        if tuple(state.shape) != tuple(machine.state.shape):
            raise SystemExit(
                f"Saved state shape {state.shape} does not match this run {machine.state.shape}."
            )
        machine.state = state
        progress_path = RESULTS / "hierarchical_progress.json"
        if progress_path.exists():
            stored = json.loads(progress_path.read_text(encoding="utf-8"))
            history = list(stored.get("epoch_history") or [])
        best_epoch = int(saved.get("best_epoch") or 1)
        best_accuracy = float(saved["validation_accuracy"])
        best_train = float(saved.get("train_accuracy") or 0.0)
        best_state = machine.state.copy()
        last_finished = max((int(row["epoch"]) for row in history), default=best_epoch)
        first_epoch = last_finished + 1
        last_epoch = last_finished + args.continue_epochs
        print(
            f"Continuing from the saved weights of epoch {best_epoch}, "
            f"validation accuracy {best_accuracy:.3f}. "
            f"Training epochs {first_epoch} to {last_epoch}.",
            flush=True,
        )

    for epoch in range(first_epoch, last_epoch + 1):
        machine.fit(upper_train, y_train, epochs=1)
        train_accuracy = float((machine.predict(upper_train) == y_train).mean())
        predictions = machine.predict(upper_val)
        val_accuracy = float((predictions == y_val).mean())
        history.append({"epoch": epoch, "train": train_accuracy, "validation": val_accuracy})
        print(f"epoch {epoch}: train {train_accuracy:.3f}, validation {val_accuracy:.3f}", flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_state = machine.state.copy()
            best_predictions = predictions.copy()
            best_train = train_accuracy
        progress = {
            "run_complete": False,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "validation": round(val_accuracy, 3),
            "best_validation": round(best_accuracy, 3),
            "validation_accuracy": best_accuracy,
            "epoch_history": history,
        }
        (RESULTS / "hierarchical_progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")

    machine.state = best_state
    if best_predictions is None:
        best_predictions = machine.predict(upper_val)
        best_train = float((machine.predict(upper_train) == y_train).mean())
    matrix = confusion_matrix(y_val, best_predictions)
    summary = {
        "stage": "hierarchical",
        "implementation": (
            "Two Tsetlin Machines. The first is the saved vanilla machine. "
            "The second is trained on that machine's clause outputs and on one bit per word "
            "for the word being present in both posts. s = 1.0 and boost is on for the second level. "
            "PyHierarchicalTsetlinMachineCUDA was not run; this computer has no NVIDIA GPU."
        ),
        "hierarchy": (
            f"vanilla clauses {vanilla.n_clauses} per class, then {args.clauses} second-level clauses"
        ),
        "data_dir": str(data_dir),
        "test_pairs_used": False,
        "train_shape": list(x_train.shape),
        "validation_shape": list(x_val.shape),
        "train_accuracy": best_train,
        "validation_accuracy": best_accuracy,
        "vanilla_validation_accuracy": vanilla_val,
        "run_complete": True,
        "best_epoch": best_epoch,
        "validation_confusion_matrix": {
            "rows": "true Y=0, true Y=1",
            "columns": "predicted Y=0, predicted Y=1",
            "values": matrix,
        },
        "hyperparameters": {
            "clauses_per_class": args.clauses,
            "T": args.T,
            "s": args.s,
            "boost_true_positive_feedback": args.boost,
            "epochs": last_epoch,
            "seed": args.seed,
            "vanilla_clauses": int(vanilla.n_clauses),
            "shared_word_bits": x_train.shape[1] // 2,
        },
    }
    clause_text = (
        "Level 1 is the saved vanilla Tsetlin Machine.\n"
        "Level 2 is a Tsetlin Machine trained on those clause outputs and on shared words.\n"
        f"s = 1.0, boost {args.boost}, best epoch {best_epoch}, validation {best_accuracy:.3f}.\n"
    )
    (RESULTS / "hierarchical.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS / "hierarchical_clauses.txt").write_text(clause_text, encoding="utf-8")
    np.savez_compressed(RESULTS / "hierarchical_state.npz", state=best_state)
    progress["run_complete"] = True
    progress["validation_accuracy"] = best_accuracy
    (RESULTS / "hierarchical_progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    print(f"Best validation accuracy {best_accuracy:.3f} at epoch {best_epoch}")
    print(f"Validation confusion matrix {matrix}")
    print(f"Wrote {RESULTS / 'hierarchical.json'}")


if __name__ == "__main__":
    main()
