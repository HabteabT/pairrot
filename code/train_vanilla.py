"""Train the vanilla Tsetlin Machine on prepared Pairrot pairs.

The model reads X and y only. Category names stay in the pair CSV files.
Validation accuracy is the pilot result. This script never reads the test
arrays.

``--boost 1`` turns on ``boost_true_positive_feedback``, the default in tmu and
pyTsetlinMachine. With it, Type I always reinforces Include on the true
literals of a firing clause, even at s = 1.0. The default here stays 0, the
setting the report describes.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    for epoch in range(1, args.epochs + 1):
        machine.fit(x_train, y_train, epochs=1)
        train_now = accuracy(y_train, machine.predict(x_train))
        print(f"  epoch {epoch}: train accuracy {train_now:.3f}")

    train_predictions = machine.predict(x_train)
    val_predictions = machine.predict(x_val)
    train_accuracy = accuracy(y_train, train_predictions)
    val_accuracy = accuracy(y_val, val_predictions)
    val_matrix = confusion_matrix(y_val, val_predictions)
    print(f"Pilot train accuracy {train_accuracy:.3f}")
    print(f"Pilot validation accuracy {val_accuracy:.3f}")
    print(f"Validation confusion matrix [true 0, true 1] x [pred 0, pred 1]: {val_matrix}")

    with (data_dir / "pairs_val.csv").open(encoding="utf-8", newline="") as handle:
        pair_ids = [row["pair_id"] for row in csv.DictReader(handle)]
    correct_index = next(
        (index for index, truth in enumerate(y_val) if int(truth) == int(val_predictions[index])),
        0,
    )
    try:
        incorrect_index = next(
            index for index, truth in enumerate(y_val) if int(truth) != int(val_predictions[index])
        )
    except StopIteration:
        incorrect_index = None

    def pair_id(index: int) -> str:
        return pair_ids[index]

    sections = [
        describe_example(
            machine,
            x_val,
            y_val,
            val_predictions,
            correct_index,
            vocabulary,
            document_frequency,
            pair_id(correct_index),
        )
    ]
    if incorrect_index is not None:
        sections.append(
            describe_example(
                machine,
                x_val,
                y_val,
                val_predictions,
                incorrect_index,
                vocabulary,
                document_frequency,
                pair_id(incorrect_index),
            )
        )
    sections.extend(clause_summary(machine, vocabulary, document_frequency))
    clause_text = "\n\n".join(sections)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "vanilla_clauses.txt").write_text(clause_text, encoding="utf-8")
    np.savez_compressed(RESULTS / "vanilla_state.npz", state=machine.state)
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
            f"Pilot validation accuracy {val_accuracy:.3f}. "
            "Test pairs were not used. Categories were not model inputs."
        ),
    }
    (RESULTS / "vanilla_pilot.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS / "vanilla_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(clause_text[:4000])
    print(f"Wrote {RESULTS / 'vanilla_pilot.json'}")
    print(f"Wrote {RESULTS / 'vanilla_clauses.txt'}")


if __name__ == "__main__":
    main()
