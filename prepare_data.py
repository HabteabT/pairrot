"""Prepare balanced document pairs and binary word features for Pairrot."""

import argparse
import csv
from itertools import combinations_with_replacement
import json
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.sparse import hstack, save_npz
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


REMOVE = ("headers", "footers", "quotes")


def sample_pairs(categories, count, rng):
    """Sample uniformly within each target class, without replacement."""
    if count < 2 or count % 2:
        raise ValueError("Pair count must be a positive even number (at least 2).")

    groups = [np.flatnonzero(categories == label) for label in np.unique(categories)]
    blocks = {0: [], 1: []}
    for i, j in combinations_with_replacement(range(len(groups)), 2):
        left, right = groups[i], groups[j]
        same = i == j
        capacity = len(left) * (len(left) - 1) // 2 if same else len(left) * len(right)
        if capacity:
            blocks[int(same)].append((left, right, capacity))

    pairs = np.empty((count, 2), dtype=np.int64)
    targets = np.repeat([0, 1], count // 2).astype(np.uint8)
    for target, target_blocks in blocks.items():
        ends = np.cumsum([block[2] for block in target_blocks], dtype=np.int64)
        available = int(ends[-1]) if ends.size else 0
        if count // 2 > available:
            raise ValueError(
                f"Target {target} has {available:,} unique pairs available, "
                f"but {count // 2:,} were requested."
            )

        # Each rank identifies exactly one unordered pair.
        ranks = rng.choice(available, count // 2, replace=False)
        block_ids = np.searchsorted(ends, ranks, side="right")
        for block_id in np.unique(block_ids):
            rows = np.flatnonzero(block_ids == block_id)
            left, right, _ = target_blocks[block_id]
            offset = ends[block_id - 1] if block_id > 0 else 0
            local_ranks = ranks[rows] - offset
            if target == 1:
                # For n posts, row i has n-i-1 possible partners after it.
                starts = np.r_[0, np.cumsum(np.arange(len(left) - 1, 0, -1))]
                a = np.searchsorted(starts, local_ranks, side="right") - 1
                b = a + 1 + local_ranks - starts[a]
            else:
                a, b = np.divmod(local_ranks, len(right))
            pairs[rows + target * (count // 2)] = np.column_stack((left[a], right[b]))

    # The target is symmetric. Randomize which post occupies the A/B slot.
    swap = rng.integers(0, 2, size=count).astype(bool)
    pairs[swap] = pairs[swap, ::-1]
    order = rng.permutation(count)
    return pairs[order], targets[order]


def verify_pairs(pairs, targets, categories, features, vocabulary_size):
    left, right = pairs.T
    if np.any(left == right) or len(np.unique(np.sort(pairs, axis=1), axis=0)) != len(pairs):
        raise RuntimeError("Found a self-pair or duplicate unordered pair.")
    if not np.array_equal(targets, categories[left] == categories[right]):
        raise RuntimeError("Found an incorrect pair target.")
    if np.count_nonzero(targets) * 2 != len(pairs):
        raise RuntimeError("Pair targets are unbalanced.")
    if features.shape != (len(pairs), 2 * vocabulary_size) or np.any(features.data != 1):
        raise RuntimeError("Expected two equally sized binary word feature blocks.")


def document_ids(subset, filenames):
    return np.array([
        f"{subset}/{Path(filename).parent.name}/{Path(filename).name}"
        for filename in filenames
    ])


def prepare(args):
    if args.vocabulary_size < 1:
        raise ValueError("Vocabulary size must be positive.")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")
    if args.seed < 0 or args.seed >= 2**32:
        raise ValueError("Seed must be in the range 0 to 2**32 - 1.")

    train = fetch_20newsgroups(
        subset="train", shuffle=False, remove=REMOVE, data_home=args.data_home
    )
    test = fetch_20newsgroups(
        subset="test", shuffle=False, remove=REMOVE, data_home=args.data_home
    )
    train_rows, validation_rows = train_test_split(
        np.arange(len(train.data)),
        test_size=args.validation_fraction,
        random_state=args.seed,
        stratify=train.target,
    )

    official_train_ids = document_ids("train", train.filenames)
    test_ids = document_ids("test", test.filenames)
    splits = {
        "train": (train, train_rows, official_train_ids[train_rows], args.train_pairs),
        "validation": (
            train, validation_rows, official_train_ids[validation_rows], args.validation_pairs
        ),
        "test": (test, np.arange(len(test.data)), test_ids, args.test_pairs),
    }
    all_ids = np.concatenate([split[2] for split in splits.values()])
    if len(np.unique(all_ids)) != len(all_ids):
        raise RuntimeError("Document IDs must be unique across the splits.")

    vectorizer = CountVectorizer(
        binary=True, max_features=args.vocabulary_size, stop_words="english", dtype=np.uint8
    )
    vectorizer.fit(train.data[i] for i in train_rows)
    vocabulary = vectorizer.get_feature_names_out().tolist()
    metadata = {
        "dataset": "20 Newsgroups (scikit-learn official train and test subsets)",
        "remove": list(REMOVE),
        "stop_words": "english",
        "vocabulary_limit": args.vocabulary_size,
        "vocabulary": vocabulary,
        "feature_order": ["post_a", "post_b"],
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "category_names": train.target_names,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "splits": {},
    }

    prepared = {}
    seeds = np.random.SeedSequence(args.seed).spawn(len(splits))
    for (name, (dataset, rows, ids, count)), seed in zip(splits.items(), seeds):
        categories = dataset.target[rows]
        pairs, targets = sample_pairs(categories, count, np.random.default_rng(seed))
        words = vectorizer.transform(dataset.data[i] for i in rows)
        features = hstack((words[pairs[:, 0]], words[pairs[:, 1]]), format="csr")
        verify_pairs(pairs, targets, categories, features, len(vocabulary))
        prepared[name] = (ids, categories, dataset.target_names, pairs, targets, features)
        metadata["splits"][name] = {
            "posts": len(rows),
            "posts_with_no_vocabulary_words": int(np.count_nonzero(words.getnnz(axis=1) == 0)),
            "pairs": count,
            "same": count // 2,
            "different": count // 2,
            "feature_columns": features.shape[1],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, (ids, categories, names, pairs, targets, features) in prepared.items():
        save_npz(args.output_dir / f"{name}_features.npz", features)
        np.save(args.output_dir / f"{name}_targets.npy", targets)
        with (args.output_dir / f"{name}_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["post_a_id", "post_b_id", "post_a_category", "post_b_category", "target"])
            writer.writerows(
                (ids[a], ids[b], names[categories[a]], names[categories[b]], int(target))
                for (a, b), target in zip(pairs, targets)
            )
        print(f"{name}: {len(ids):,} posts, {len(pairs):,} pairs, {features.shape[1]:,} features")
        if name == "train":
            for target in (1, 0):
                a, b = pairs[np.flatnonzero(targets == target)[0]]
                print(f"  {names[categories[a]]} + {names[categories[b]]} -> Y={target}")

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved verified dataset to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=Path(".cache/20newsgroups"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--vocabulary-size", type=int, default=1000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-pairs", type=int, default=10000)
    parser.add_argument("--validation-pairs", type=int, default=2000)
    parser.add_argument("--test-pairs", type=int, default=4000)
    args = parser.parse_args()
    try:
        prepare(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
