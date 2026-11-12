"""Train the vanilla Tsetlin Machine with the course library on prepared Pairrot pairs.

The course asks for the vanilla machine to be built with
``PyHierarchicalTsetlinMachineCUDA``, following ``examples/MultiConceptDemo.py``
and this hierarchy structure, which is one flat AND over all features:

    hierarchy_structure=((tm.AND_GROUP, features), (tm.AND_GROUP, 1))

This script does that. It reads the same ``features_train.npz`` and
``features_val.npz`` as ``train_vanilla.py`` and never reads the test file.

The library needs an NVIDIA GPU, CUDA, and PyCUDA. On a computer without one,
run it in Google Colab with a GPU runtime (see ``code/README.md``).

Two settings differ between the course example and the library defaults:

- ``MultiConceptDemo.py`` passes ``boost_true_positive_feedback=0`` and
  ``append_negated=False``. Its task uses s = 21.1.
- The library defaults are ``boost_true_positive_feedback=1`` and
  ``append_negated=True``.

At s = 1.0 the library's Type I feedback includes a true literal with
probability 1 - 1/s = 0 when boost is 0, and negated literals exist only when
``append_negated`` is on. Reasoning by elimination needs both, so this script
defaults to the library defaults. ``--boost 0`` and ``--no-negated`` give the
example's settings.

This script was written without a GPU to run it on. The first run on a GPU is
its test.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "dataset" / "prepared" / "s457_f300_t30000_v2000_e2000_val0p2_fulltext_topical"
RESULTS = Path(__file__).resolve().parent / "results"


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir / f"features_{split}.npz"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run dataset/prepare_data.py first.")
    stored = np.load(path)
    # The library reads unsigned 32-bit inputs, as in MultiConceptDemo.py.
    return stored["X"].astype(np.uint32), stored["y"].astype(np.uint32)


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((2, 2), dtype=int)
    for truth, prediction in zip(targets, predictions):
        matrix[int(truth), int(prediction)] += 1
    return matrix.tolist()


def included_literals(machine) -> np.ndarray:
    """Include action of every literal of every clause of one class machine.

    Reads the automaton states from the GPU once. ``machine.ta_action`` copies
    the whole state for every single literal, which is too slow for a listing.
    The layout is the one ``ta_action`` uses: the highest state bit is Include.
    """

    import pycuda.driver as cuda

    shape = (
        machine.number_of_clauses,
        machine.number_of_ta_teams,
        machine.number_of_literal_chunks_per_leaf,
        machine.number_of_state_bits,
    )
    state = np.empty(int(np.prod(shape)), dtype=np.uint32)
    cuda.memcpy_dtoh(state, machine.ta_state_hierarchy_gpu)
    include_bits = state.reshape(shape)[:, 0, :, machine.number_of_state_bits - 1]
    literals = machine.number_of_literals_per_leaf
    index = np.arange(literals)
    return ((include_bits[:, index // 32] >> (index % 32)) & 1).astype(bool)


def literal_phrase(literal: int, n_features: int, vocabulary: list[str], negated_appended: bool) -> str:
    """Feature layout is Post A then Post B. Negated literals follow the plain ones."""

    negated = negated_appended and literal >= n_features
    feature = literal - n_features if negated else literal
    size = len(vocabulary)
    post = "Post A" if feature < size else "Post B"
    word = vocabulary[feature % size]
    return f"{word} absent from {post}" if negated else f"{word} present in {post}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vanilla Tsetlin Machine with PyHierarchicalTsetlinMachineCUDA on prepared pairs."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--clauses", type=int, default=800, help="Clauses per class.")
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--s", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--state-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=457)
    parser.add_argument("--boost", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--no-negated",
        action="store_true",
        help="append_negated=False, as in MultiConceptDemo.py. Clauses then have no negated literals.",
    )
    parser.add_argument("--clauses-shown", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import PyHierarchicalTsetlinMachineCUDA.tm as tm
        from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
    except Exception as error:
        raise SystemExit(
            "Could not load PyHierarchicalTsetlinMachineCUDA "
            f"({type(error).__name__}: {error}). It needs an NVIDIA GPU with CUDA and PyCUDA. "
            "Install it with: pip install PyHierarchicalTsetlinMachineCUDA"
        )

    data_dir = args.data.resolve()
    x_train, y_train = load_split(data_dir, "train")
    x_val, y_val = load_split(data_dir, "val")
    features = x_train.shape[1]
    vocabulary = json.loads((data_dir / "vocabulary.json").read_text(encoding="utf-8"))["words"]
    print(f"Train X {tuple(x_train.shape)}, validation X {tuple(x_val.shape)}. Test pairs are not read.")

    # The seed argument seeds the automata. The library also uses numpy's
    # global generator when it balances and shuffles the examples.
    np.random.seed(args.seed)
    machine = MultiClassTsetlinMachine(
        args.clauses,
        args.T,
        args.s,
        seed=args.seed,
        number_of_state_bits=args.state_bits,
        boost_true_positive_feedback=args.boost,
        append_negated=not args.no_negated,
        # The vanilla structure given for the project: one flat AND over all features.
        hierarchy_structure=(
            (tm.AND_GROUP, features),
            (tm.AND_GROUP, 1),
        ),
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        machine.fit(x_train, y_train)
        val_now = float((machine.predict(x_val) == y_val).mean())
        history.append(val_now)
        print(f"  epoch {epoch}: validation accuracy {val_now:.3f}", flush=True)

    train_predictions = machine.predict(x_train)
    val_predictions = machine.predict(x_val)
    train_accuracy = float((train_predictions == y_train).mean())
    val_accuracy = float((val_predictions == y_val).mean())
    val_matrix = confusion_matrix(y_val, val_predictions)
    print(f"Train accuracy {train_accuracy:.3f}")
    print(f"Validation accuracy {val_accuracy:.3f}")
    print(f"Validation confusion matrix [true 0, true 1] x [pred 0, pred 1]: {val_matrix}")

    lines = []
    totals = {"included": 0, "negated": 0}
    for class_id, class_machine in enumerate(machine.tms):
        included = included_literals(class_machine)
        totals["included"] += int(included.sum())
        if not args.no_negated:
            totals["negated"] += int(included[:, features:].sum())
        shown = 0
        for clause_id in range(included.shape[0]):
            chosen = np.flatnonzero(included[clause_id])
            if len(chosen) == 0 or shown >= args.clauses_shown:
                continue
            shown += 1
            phrases = [
                literal_phrase(int(index), features, vocabulary, not args.no_negated) for index in chosen[:8]
            ]
            more = f", and {len(chosen) - 8} more" if len(chosen) > 8 else ""
            weight = int(class_machine.get_clause_weight(clause_id))
            vote = "votes for" if weight >= 0 else "votes against"
            lines.append(f"class {class_id} clause {clause_id} ({vote}): {', '.join(phrases)}{more}")
    header = f"{totals['included']} included literals, {totals['negated']} of them negated."
    clause_text = "\n".join([header, *lines])
    print(clause_text)

    with (data_dir / "pairs_val.csv").open(encoding="utf-8", newline="") as handle:
        n_pairs = sum(1 for _ in csv.DictReader(handle))
    summary = {
        "stage": "vanilla-cuda",
        "library": "PyHierarchicalTsetlinMachineCUDA",
        "hierarchy_structure": f"((AND_GROUP, {features}), (AND_GROUP, 1))",
        "data_dir": str(data_dir),
        "test_pairs_used": False,
        "train_shape": list(x_train.shape),
        "validation_shape": list(x_val.shape),
        "validation_pairs_in_csv": n_pairs,
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "validation_accuracy_per_epoch": history,
        "validation_confusion_matrix": {
            "rows": "true Y=0, true Y=1",
            "columns": "predicted Y=0, predicted Y=1",
            "values": val_matrix,
        },
        "included_literals": totals,
        "hyperparameters": {
            "clauses_per_class": args.clauses,
            "T": args.T,
            "s": args.s,
            "epochs": args.epochs,
            "number_of_state_bits": args.state_bits,
            "boost_true_positive_feedback": args.boost,
            "append_negated": not args.no_negated,
            "seed": args.seed,
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "vanilla_cuda.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS / "vanilla_cuda_clauses.txt").write_text(clause_text, encoding="utf-8")
    print(f"Wrote {RESULTS / 'vanilla_cuda.json'}")


if __name__ == "__main__":
    main()
