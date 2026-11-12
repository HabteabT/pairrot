"""Train the vanilla Tsetlin Machine on prepared Pairrot pairs.

The model reads X and y only. Category names stay in the pair CSV files.
Validation accuracy is the pilot result. This script never reads the test
arrays.

The model is saved after every epoch that improves validation accuracy, with
its own confusion matrix and clause list, so a run that is stopped early still
leaves a model that matches a reported number. The previous saved model is
moved to ``results/archive`` first and is never overwritten.

``--boost 1`` turns on ``boost_true_positive_feedback``, the default in tmu and
pyTsetlinMachine. With it, Type I always reinforces Include on the true
literals of a firing clause, even at s = 1.0. The default here stays 0, the
setting the report describes.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from vanilla_tsetlin import VanillaTsetlinMachine, literal_phrase, verify_s_equals_one


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "dataset" / "prepared" / "s457_f1000_t1000_v200_e200_val0p2"
RESULTS = Path(__file__).resolve().parent / "results"


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one feature file. The category table is intentionally not read."""

    path = data_dir / f"features_{split}.npz"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run dataset/prepare_data.py first.")
    stored = np.load(path)
    if set(stored.files) != {"X", "y"}:
        raise SystemExit(f"{path.name} should contain only X and y. Found {stored.files}.")
    features = stored["X"].astype(np.uint8)
    targets = stored["y"].astype(np.uint32)
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        raise SystemExit(f"{path.name} has an unexpected shape: X {features.shape}, y {targets.shape}.")
    if not np.isin(features, (0, 1)).all():
        raise SystemExit(f"{path.name} contains values other than 0 and 1.")
    return features, targets


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((2, 2), dtype=int)
    for truth, prediction in zip(targets, predictions):
        matrix[int(truth), int(prediction)] += 1
    return matrix.tolist()


def accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float((targets == predictions).mean())


def learn_one_bit() -> float:
    """Illustrative check: the label is the first bit, and the other bits are noise."""

    draw = np.random.default_rng(11)
    features = draw.integers(0, 2, size=(300, 6), dtype=np.uint8)
    targets = features[:, 0].astype(np.uint32)
    machine = VanillaTsetlinMachine(n_features=6, n_clauses=10, T=8, s=1.0, seed=11)
    machine.fit(features, targets, epochs=12)
    return accuracy(targets, machine.predict(features))


def topical_literals(
    indexes: np.ndarray,
    n_features: int,
    vocabulary: list[str],
    document_frequency: list[int],
    limit: int,
) -> list[str]:
    vocabulary_size = len(vocabulary)

    def frequency(literal_index: int) -> int:
        feature = literal_index - n_features if literal_index >= n_features else literal_index
        word_index = feature - vocabulary_size if feature >= vocabulary_size else feature
        return document_frequency[word_index]

    ordered = sorted(indexes.tolist(), key=frequency)
    return [literal_phrase(index, n_features, vocabulary) for index in ordered[:limit]]


def describe_example(
    machine: VanillaTsetlinMachine,
    features: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    index: int,
    vocabulary: list[str],
    document_frequency: list[int],
    pair_id: str,
) -> str:
    scores, outputs = machine.predict_scores(features[index])
    lines = [
        f"Pair {pair_id}",
        f"  true Y = {int(targets[index])}, predicted Y = {int(predictions[index])}",
        f"  clipped votes: class 0 = {int(scores[0])}, class 1 = {int(scores[1])}",
        "  firing clauses during prediction:",
    ]
    any_firing = False
    for class_id, output in enumerate(outputs):
        for clause_id in np.flatnonzero(output):
            any_firing = True
            included = machine.included_indices(class_id, clause_id)
            polarity = "positive" if clause_id % 2 == 0 else "negative"
            sample = topical_literals(
                included,
                machine.n_features,
                vocabulary,
                document_frequency,
                limit=6,
            )
            negated = int((included >= machine.n_features).sum())
            lines.append(
                f"    class {class_id} clause {clause_id} ({polarity}): "
                f"{len(included)} literals, {negated} negated. "
                f"Least common: {', '.join(sample) if sample else '(none)'}"
            )
    if not any_firing:
        lines.append("    (none)")
    return "\n".join(lines)


def clause_summary(
    machine: VanillaTsetlinMachine,
    vocabulary: list[str],
    document_frequency: list[int],
) -> list[str]:
    lines = ["Included literals after training. Least common words are shown first."]
    negated_total = 0
    included_total = 0
    for class_id in range(machine.n_classes):
        for clause_id in range(machine.n_clauses):
            included = machine.included_indices(class_id, clause_id)
            if len(included) == 0:
                continue
            included_total += len(included)
            negated_total += int((included >= machine.n_features).sum())
            polarity = "positive" if clause_id % 2 == 0 else "negative"
            sample = topical_literals(
                included,
                machine.n_features,
                vocabulary,
                document_frequency,
                limit=8,
            )
            lines.append(
                f"class {class_id} clause {clause_id} ({polarity}): "
                f"{len(included)} literals, {int((included >= machine.n_features).sum())} negated. "
                f"{', '.join(sample)}"
            )
    if included_total == 0:
        lines.append("No clause included a literal.")
    else:
        lines.insert(
            1,
            f"{included_total} included literal slots, {negated_total} of them negated.",
        )
    return lines


SAVED_FILES = ("vanilla_state.npz", "vanilla_pilot.json", "vanilla_status.json", "vanilla_clauses.txt")


def _replace_file(path: Path, write) -> None:
    """Write to a temporary neighbour, then swap it in."""

    temporary = path.with_name(path.name + ".partial")
    write(temporary)
    temporary.replace(path)


def _save_state(path: Path, state: np.ndarray) -> None:
    # numpy adds ".npz" to a name without it, so write through an open file.
    with path.open("wb") as handle:
        np.savez_compressed(handle, state=state)


def _update_history(history: list[dict], complete: bool) -> None:
    """Record finished epochs without touching the saved weights."""

    for name in ("vanilla_pilot.json", "vanilla_status.json"):
        path = RESULTS / name
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["history"] = history
        summary["epochs_finished"] = len(history)
        summary["run_complete"] = complete
        payload = json.dumps(summary, indent=2)
        _replace_file(path, lambda target: target.write_text(payload, encoding="utf-8"))


def archive_previous_results() -> Path | None:
    """Move the last saved model aside before a new run writes its own.

    A new run used to overwrite the old files, so a good run could be lost to
    a later, worse one. The old files now go to ``results/archive/<time>_val<accuracy>``.
    """

    existing = [RESULTS / name for name in SAVED_FILES if (RESULTS / name).exists()]
    if not existing:
        return None
    label = "unknown"
    pilot = RESULTS / "vanilla_pilot.json"
    if pilot.exists():
        try:
            label = f"{json.loads(pilot.read_text(encoding='utf-8'))['validation_accuracy']:.3f}"
        except (ValueError, KeyError, TypeError):
            pass
    target = RESULTS / "archive" / f"{datetime.now():%Y%m%d-%H%M%S}_val{label}"
    target.mkdir(parents=True, exist_ok=False)
    for path in existing:
        path.replace(target / path.name)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the vanilla Tsetlin Machine on prepared pairs.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--clauses", type=int, default=200)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--s", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--state-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=457)
    parser.add_argument(
        "--boost",
        type=int,
        choices=(0, 1),
        default=0,
        help="boost_true_positive_feedback. 0 is the report setting. 1 is the tmu/pyTsetlinMachine default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Feedback checks for s=1.0:")
    for check in verify_s_equals_one():
        print(f"  PASS {check}")
    toy_accuracy = learn_one_bit()
    print(f"Illustrative one-bit training accuracy: {toy_accuracy:.3f}")
    if toy_accuracy < 0.8:
        raise SystemExit("The illustrative one-bit problem stayed below 0.80. The update loop needs another look.")

    data_dir = args.data.resolve()
    x_train, y_train = load_split(data_dir, "train")
    x_val, y_val = load_split(data_dir, "val")
    vocabulary_path = data_dir / "vocabulary.json"
    if not vocabulary_path.exists():
        raise SystemExit(f"Missing {vocabulary_path}. Run dataset/prepare_data.py first.")
    vocabulary_payload = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    vocabulary = vocabulary_payload["words"]
    document_frequency = vocabulary_payload["document_frequency"]
    print(f"Train X {tuple(x_train.shape)}, validation X {tuple(x_val.shape)}. Categories were not loaded.")

    machine = VanillaTsetlinMachine(
        n_features=x_train.shape[1],
        n_clauses=args.clauses,
        T=args.T,
        s=args.s,
        n_classes=2,
        number_of_state_bits=args.state_bits,
        seed=args.seed,
        boost_true_positive_feedback=args.boost,
    )
    with (data_dir / "pairs_val.csv").open(encoding="utf-8", newline="") as handle:
        pair_ids = [row["pair_id"] for row in csv.DictReader(handle)]

    def save(epoch: int, history: list[dict], complete: bool) -> float:
        """Write the model as it stands now, with its own scores, matrix, and clauses.

        Every file is written under a temporary name and then renamed, so a
        run that is stopped in the middle of a save leaves the previous files whole.
        """

        train_predictions = machine.predict(x_train)
        val_predictions = machine.predict(x_val)
        train_accuracy = accuracy(y_train, train_predictions)
        val_accuracy = accuracy(y_val, val_predictions)
        val_matrix = confusion_matrix(y_val, val_predictions)

        correct_index = next(
            (index for index, truth in enumerate(y_val) if int(truth) == int(val_predictions[index])),
            0,
        )
        incorrect_index = next(
            (index for index, truth in enumerate(y_val) if int(truth) != int(val_predictions[index])),
            None,
        )
        sections = [
            describe_example(
                machine, x_val, y_val, val_predictions, index, vocabulary, document_frequency, pair_ids[index]
            )
            for index in (correct_index, incorrect_index)
            if index is not None
        ]
        sections.extend(clause_summary(machine, vocabulary, document_frequency))
        clause_text = "\n\n".join(sections)

        summary = {
            "stage": "vanilla-pilot",
            "label": "pilot, not the final test comparison",
            "data_dir": str(data_dir),
            "test_pairs_used": False,
            "train_shape": list(x_train.shape),
            "validation_shape": list(x_val.shape),
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "accuracy": val_accuracy,
            "validation_confusion_matrix": {
                "rows": "true Y=0, true Y=1",
                "columns": "predicted Y=0, predicted Y=1",
                "values": val_matrix,
            },
            "saved_epoch": epoch,
            "epochs_finished": len(history),
            "epochs_requested": args.epochs,
            "run_complete": complete,
            "saved_model_is": "the epoch with the highest validation accuracy so far",
            "history": history,
            "illustrative_one_bit_accuracy": toy_accuracy,
            "hyperparameters": {
                "clauses_per_class": args.clauses,
                "T": args.T,
                "s": args.s,
                "epochs": args.epochs,
                "number_of_state_bits": args.state_bits,
                "boost_true_positive_feedback": args.boost,
                "weighted_clauses": False,
                "seed": args.seed,
            },
            "implementation": {
                "library_attempted": "tmu==0.7.1",
                "library_error": "Microsoft Visual C++ 14.0 or greater is required",
                "feedback_source": (
                    "pyTsetlinMachine ConvolutionalTsetlinMachine.c, "
                    f"boost_true_positive_feedback {args.boost}"
                ),
                "s_equals_one": (
                    "Type I include probability is 1 on true literals of a firing clause (boost on), "
                    "and Type I exclude probability is 1"
                    if args.boost
                    else "Type I include probability is 0 and Type I exclude probability is 1"
                ),
            },
            "note": (
                f"Validation accuracy {val_accuracy:.3f} at epoch {epoch}. "
                "Test pairs were not used. Categories were not model inputs."
            ),
        }
        payload = json.dumps(summary, indent=2)
        _replace_file(RESULTS / "vanilla_clauses.txt", lambda path: path.write_text(clause_text, encoding="utf-8"))
        _replace_file(RESULTS / "vanilla_state.npz", lambda path: _save_state(path, machine.state))
        _replace_file(RESULTS / "vanilla_pilot.json", lambda path: path.write_text(payload, encoding="utf-8"))
        _replace_file(RESULTS / "vanilla_status.json", lambda path: path.write_text(payload, encoding="utf-8"))
        return val_accuracy

    RESULTS.mkdir(parents=True, exist_ok=True)
    archived = archive_previous_results()
    if archived is not None:
        print(f"The previous saved model was moved to {archived}. Nothing was deleted.")

    history: list[dict] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state = machine.state.copy()
    try:
        for epoch in range(1, args.epochs + 1):
            machine.fit(x_train, y_train, epochs=1)
            train_now = accuracy(y_train, machine.predict(x_train))
            val_now = accuracy(y_val, machine.predict(x_val))
            history.append({"epoch": epoch, "train_accuracy": train_now, "validation_accuracy": val_now})
            improved = val_now > best_accuracy
            if improved:
                best_accuracy, best_epoch = val_now, epoch
                best_state = machine.state.copy()
                save(epoch, history, complete=False)
            else:
                # Keep the saved model, but record that another epoch has finished.
                _update_history(history, complete=False)
            print(
                f"  epoch {epoch}: train accuracy {train_now:.3f}, validation accuracy {val_now:.3f}"
                + ("  (saved)" if improved else ""),
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nStopped by the keyboard. The best epoch so far is already saved.")

    if best_epoch == 0:
        raise SystemExit("No epoch finished, so no model was saved.")
    # Finish with the best epoch in memory, so the files and the printout describe the same model.
    machine.state = best_state
    final_accuracy = save(best_epoch, history, complete=len(history) == args.epochs)
    summary = json.loads((RESULTS / "vanilla_pilot.json").read_text(encoding="utf-8"))
    print(f"Saved model: epoch {best_epoch} of {len(history)} finished")
    print(f"Pilot train accuracy {summary['train_accuracy']:.3f}")
    print(f"Pilot validation accuracy {final_accuracy:.3f}")
    print(
        "Validation confusion matrix [true 0, true 1] x [pred 0, pred 1]: "
        f"{summary['validation_confusion_matrix']['values']}"
    )
    print((RESULTS / "vanilla_clauses.txt").read_text(encoding="utf-8")[:4000])
    print(f"Wrote {RESULTS / 'vanilla_pilot.json'}")
    print(f"Wrote {RESULTS / 'vanilla_clauses.txt'}")


if __name__ == "__main__":
    main()
