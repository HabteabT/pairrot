"""Build the Pairrot dataset: pairs of 20 Newsgroups posts.

Assignment requirements implemented here
----------------------------------------
- Use the 20 Newsgroups dataset and all 20 categories.
- Load the official training posts and the official test posts.
- Each example is a pair of two distinct posts.
- Y = 1 when the two posts belong to the same category.
- Y = 0 when they belong to different categories.
- Remove headers, footers, and quoted text with the dataset loader.
- Represent each post with binary word features from one shared vocabulary.
- Post A occupies the first feature block and Post B the second.
- A feature is 1 when that word is present and 0 when it is absent.
- Category names are used only to build and check Y. They are not features.
- Keep the official test posts out of fitting and out of the validation split.

Experiment choices, not assignment requirements
-----------------------------------------------
- Hold out 20% of the official training posts as validation.
- The validation split is stratified by category.
- The vocabulary has at most 1,000 words, selected on training posts only.
- Ranking uses document frequency. Alphabetical order breaks ties.
- A token is a lowercase letter sequence of length at least 2.
- Stop words are kept, and words are not stemmed.
- Posts that become empty after cleaning are kept.
- Pair counts are even, with equally many positive and negative pairs.
- Pairs are unordered. A reversed copy of a pair is treated as the same pair.
- Which post is A or B is chosen at random and then saved.
- The default seed is 457.

The saved ``.npz`` files contain only X and y. The CSV files keep the document
ids and category names so a person can check the pairs by hand.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets._twenty_newsgroups import ARCHIVE
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
TOKEN_PATTERN = re.compile(r"[a-z]{2,}")
PAIR_SEED_OFFSET = {"train": 1, "val": 2, "test": 3}
EXPECTED_CATEGORIES = frozenset(
    {
        "alt.atheism",
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
        "misc.forsale",
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
        "soc.religion.christian",
        "talk.politics.guns",
        "talk.politics.mideast",
        "talk.politics.misc",
        "talk.religion.misc",
    }
)

ASSIGNMENT_REQUIREMENTS = {
    "dataset": "20 Newsgroups, all 20 categories",
    "official_splits_loaded": ["train", "test"],
    "example": "two distinct posts",
    "target": "Y=1 if the posts share a category, otherwise Y=0",
    "cleaning_required_by_assignment": ["headers", "footers", "quotes"],
    "features": "one vocabulary, binary presence, separate blocks for Post A and Post B",
    "model_inputs": "binary word features only; category names are not features",
    "held_out_until_final_comparison": "official test posts",
}


@dataclass(frozen=True)
class Document:
    """One newsgroup post after the required loader cleaning."""

    doc_id: str
    category: str
    origin: str
    text: str
    tokens: frozenset[str]


def tokenize(text: str) -> frozenset[str]:
    """Return the set of word tokens in one post.

    Presence is all the feature encoding needs, so repeated words collapse
    to one token here. The pattern itself is an experiment choice.
    """

    return frozenset(TOKEN_PATTERN.findall(text.lower()))


def make_doc_id(filename: str) -> str:
    """Build a machine-independent id from the 20 Newsgroups file path.

    The cache directory differs between computers. The archive name, category,
    and message number do not, and together they identify one file.
    """

    path = Path(filename)
    archive = path.parent.parent.name
    category = path.parent.name
    message_id = path.name
    if not archive or not category or not message_id:
        raise RuntimeError(f"Could not build a document id from {filename!r}.")
    return f"{archive}/{category}/{message_id}"


# scikit-learn's packaged figshare URL returned HTTP 403 in this environment.
# These are the original by-date locations named in sklearn's loader source.
ORIGINAL_ARCHIVE_URLS = (
    "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz",
    "https://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 256), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "pairrot-ikt457"})
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def ensure_newsgroups_archive(data_home: Path) -> dict[str, str]:
    """Put the by-date archive where scikit-learn looks, after checking its checksum.

    The loader skips its own download when ``20news_home/20news-bydate.tar.gz``
    already has ``ARCHIVE.checksum``. That checksum is the acceptance test:
    a mirror is used only when the bytes are the archive this scikit-learn expects.
    """

    pickled_cache = data_home / "20news-bydate_py3.pkz"
    if pickled_cache.exists():
        return {
            "status": "scikit-learn cache already present",
            "cache": str(pickled_cache),
            "checksum_expected": ARCHIVE.checksum,
        }

    archive_dir = data_home / "20news_home"
    archive_path = archive_dir / ARCHIVE.filename
    if archive_path.exists() and _sha256(archive_path) == ARCHIVE.checksum:
        return {
            "status": "local archive matched the scikit-learn checksum",
            "source": str(archive_path),
            "checksum": ARCHIVE.checksum,
        }

    archive_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for url in ORIGINAL_ARCHIVE_URLS:
        print(f"Downloading the 20 Newsgroups by-date archive from {url}")
        try:
            _download_file(url, archive_path)
        except Exception as error:
            failures.append(f"{url}: {type(error).__name__}: {error}")
            continue
        digest = _sha256(archive_path)
        if digest == ARCHIVE.checksum:
            return {"status": "downloaded", "source": url, "checksum": digest}
        failures.append(f"{url}: checksum {digest} did not match {ARCHIVE.checksum}")
        archive_path.unlink(missing_ok=True)

    raise RuntimeError(
        "Could not obtain the 20 Newsgroups by-date archive with the checksum "
        f"required by the installed scikit-learn ({ARCHIVE.checksum}). "
        + " | ".join(failures)
    )


def load_official_subset(subset: str, data_home: Path) -> tuple[list[Document], list[str]]:
    """Load one official 20 Newsgroups subset, with headers, footers, and quotes removed."""

    bunch = fetch_20newsgroups(
        data_home=str(data_home),
        subset=subset,
        categories=None,
        shuffle=False,
        remove=("headers", "footers", "quotes"),
        download_if_missing=True,
    )
    origin = "official_train" if subset == "train" else "official_test"
    documents = [
        Document(
            doc_id=make_doc_id(filename),
            category=bunch.target_names[target],
            origin=origin,
            text=text,
            tokens=tokenize(text),
        )
        for text, target, filename in zip(bunch.data, bunch.target, bunch.filenames)
    ]
    return documents, list(bunch.target_names)


def split_training_documents(
    documents: list[Document],
    val_fraction: float,
    seed: int,
) -> tuple[list[Document], list[Document]]:
    """Hold out a stratified validation set from the official training posts."""

    if not 0.0 < val_fraction < 1.0:
        raise RuntimeError(
            f"The validation fraction must be between 0 and 1. Got {val_fraction}."
        )
    labels = [document.category for document in documents]
    train_documents, val_documents = train_test_split(
        documents,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )
    return list(train_documents), list(val_documents)


def select_vocabulary(documents: list[Document], max_features: int) -> tuple[list[str], list[int]]:
    """Choose the shared vocabulary from training posts only.

    Document frequency matches the binary features: a word counts once for a
    post no matter how often it is repeated. Alphabetical order makes ties
    reproducible when two words occur in the same number of posts.
    """

    if max_features < 1:
        raise RuntimeError(f"The vocabulary limit must be at least 1. Got {max_features}.")

    frequencies: Counter[str] = Counter()
    for document in documents:
        frequencies.update(document.tokens)

    ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) < max_features:
        raise RuntimeError(
            "Impossible vocabulary request: the training posts contain "
            f"{len(ranked)} distinct tokens, which is fewer than the requested "
            f"limit of {max_features}."
        )

    chosen = ranked[:max_features]
    words = [word for word, _ in chosen]
    document_frequency = [count for _, count in chosen]
    return words, document_frequency


def _maximum_unordered_pairs(n_documents: int) -> int:
    return n_documents * (n_documents - 1) // 2


def _maximum_positive_pairs(documents: list[Document]) -> int:
    counts = Counter(document.category for document in documents)
    return sum(count * (count - 1) // 2 for count in counts.values())


def _sample_distinct_positions(rng: np.random.Generator, size: int) -> tuple[int, int]:
    first = int(rng.integers(0, size))
    second = int(rng.integers(0, size - 1))
    if second >= first:
        second += 1
    return first, second


def sample_balanced_pairs(
    documents: list[Document],
    n_pairs: int,
    seed: int,
    split: str,
) -> list[tuple[int, int]]:
    """Sample equal numbers of positive and negative unordered pairs.

    Every unordered pair of distinct posts is equally likely inside its class.
    The returned order is shuffled, and each pair is randomly oriented into
    Post A and Post B. The orientation is part of the saved example because
    the two feature blocks are different.
    """

    if n_pairs < 2 or n_pairs % 2 != 0:
        raise RuntimeError(
            f"The {split} pair count must be a positive even number so the two "
            f"classes can be equal. Got {n_pairs}."
        )

    n_positive = n_pairs // 2
    n_negative = n_pairs // 2
    n_documents = len(documents)
    max_positive = _maximum_positive_pairs(documents)
    max_negative = _maximum_unordered_pairs(n_documents) - max_positive

    if n_positive > max_positive:
        raise RuntimeError(
            f"Impossible pair request for the {split} split: asked for {n_positive} "
            f"unique positive pairs, but only {max_positive} unique unordered "
            f"same-category pairs exist among {n_documents} documents."
        )
    if n_negative > max_negative:
        raise RuntimeError(
            f"Impossible pair request for the {split} split: asked for {n_negative} "
            f"unique negative pairs, but only {max_negative} unique unordered "
            f"different-category pairs exist among {n_documents} documents."
        )

    rng = np.random.default_rng(seed)
    by_category: dict[str, list[int]] = {}
    for index, document in enumerate(documents):
        by_category.setdefault(document.category, []).append(index)

    category_names = sorted(name for name, indexes in by_category.items() if len(indexes) >= 2)
    category_weights = np.array(
        [len(by_category[name]) * (len(by_category[name]) - 1) // 2 for name in category_names],
        dtype=np.float64,
    )
    category_weights /= category_weights.sum()

    seen: set[tuple[str, str]] = set()
    positive: list[tuple[int, int]] = []
    negative: list[tuple[int, int]] = []
    attempt_limit = n_pairs * 1000 + 1000
    attempts = 0

    while len(positive) < n_positive or len(negative) < n_negative:
        attempts += 1
        if attempts > attempt_limit:
            raise RuntimeError(
                f"Stopped while sampling the {split} split after {attempts} draws. "
                f"Collected {len(positive)} unique positive pairs and {len(negative)} "
                f"unique negative pairs, but {n_positive} and {n_negative} were "
                "requested. The requested number of unique pairs was not reached."
            )

        if len(positive) < n_positive:
            category = str(rng.choice(category_names, p=category_weights))
            indexes = by_category[category]
            left_pos, right_pos = _sample_distinct_positions(rng, len(indexes))
            left = indexes[left_pos]
            right = indexes[right_pos]
            same_category = True
        else:
            left, right = _sample_distinct_positions(rng, n_documents)
            same_category = documents[left].category == documents[right].category

        key = _unordered_key(documents[left].doc_id, documents[right].doc_id)
        if key in seen:
            continue
        if same_category and len(positive) < n_positive:
            seen.add(key)
            positive.append((left, right))
        elif not same_category and len(negative) < n_negative:
            seen.add(key)
            negative.append((left, right))

    oriented = [_orient_pair(rng, left, right) for left, right in positive + negative]
    order = rng.permutation(len(oriented))
    return [oriented[int(index)] for index in order]


def _unordered_key(doc_a_id: str, doc_b_id: str) -> tuple[str, str]:
    if doc_a_id == doc_b_id:
        raise RuntimeError(f"Rejected a self-pair for document {doc_a_id!r}.")
    return (doc_a_id, doc_b_id) if doc_a_id < doc_b_id else (doc_b_id, doc_a_id)


def _orient_pair(rng: np.random.Generator, left: int, right: int) -> tuple[int, int]:
    if float(rng.random()) < 0.5:
        return left, right
    return right, left


def build_feature_matrix(
    pairs: list[tuple[int, int]],
    documents: list[Document],
    word_index: dict[str, int],
    vocabulary_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode pairs into Post A features followed by Post B features."""

    features = np.zeros((len(pairs), vocabulary_size * 2), dtype=np.uint8)
    targets = np.zeros(len(pairs), dtype=np.uint8)
    for row, (left, right) in enumerate(pairs):
        post_a = documents[left]
        post_b = documents[right]
        _set_block(features[row, :vocabulary_size], post_a.tokens, word_index)
        _set_block(features[row, vocabulary_size:], post_b.tokens, word_index)
        targets[row] = 1 if post_a.category == post_b.category else 0
    return features, targets


def _set_block(block: np.ndarray, tokens: frozenset[str], word_index: dict[str, int]) -> None:
    for token in tokens:
        feature = word_index.get(token)
        if feature is not None:
            block[feature] = 1


def recompute_feature_row(
    post_a: Document,
    post_b: Document,
    word_index: dict[str, int],
    vocabulary_size: int,
) -> np.ndarray:
    """Build one feature row again, separately from the matrix builder."""

    row = np.zeros(vocabulary_size * 2, dtype=np.uint8)
    _set_block(row[:vocabulary_size], post_a.tokens, word_index)
    _set_block(row[vocabulary_size:], post_b.tokens, word_index)
    return row


def words_in_block(block: np.ndarray, vocabulary: list[str]) -> list[str]:
    return [vocabulary[index] for index, value in enumerate(block.tolist()) if value == 1]


def preview_text(text: str, limit: int = 180) -> str:
    collapsed = " ".join(text.split())
    safe = collapsed.encode("ascii", "replace").decode("ascii")
    if len(safe) <= limit:
        return safe
    return safe[: limit - 3] + "..."


def pair_rows_for_split(
    split: str,
    pairs: list[tuple[int, int]],
    documents: list[Document],
    features: np.ndarray,
    targets: np.ndarray,
    vocabulary_size: int,
) -> list[dict[str, object]]:
    rows = []
    for index, (left, right) in enumerate(pairs):
        post_a = documents[left]
        post_b = documents[right]
        rows.append(
            {
                "pair_id": f"{split}-{index:05d}",
                "split": split,
                "doc_a_id": post_a.doc_id,
                "doc_b_id": post_b.doc_id,
                "category_a": post_a.category,
                "category_b": post_b.category,
                "y": int(targets[index]),
                "n_features_a": int(features[index, :vocabulary_size].sum()),
                "n_features_b": int(features[index, vocabulary_size:].sum()),
            }
        )
    return rows


def digest_pairs(rows: list[dict[str, object]]) -> str:
    payload = "\n".join(
        f"{row['pair_id']},{row['doc_a_id']},{row['doc_b_id']},{row['y']}" for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def document_inventory(split: str, documents: list[Document]) -> list[dict[str, object]]:
    return [
        {
            "doc_id": document.doc_id,
            "split": split,
            "origin": document.origin,
            "category": document.category,
            "n_tokens": len(document.tokens),
            "empty_after_tokenization": int(len(document.tokens) == 0),
        }
        for document in documents
    ]


def category_counts(documents: list[Document]) -> dict[str, int]:
    return dict(sorted(Counter(document.category for document in documents).items()))


def verify_prepared_data(
    splits: dict[str, list[Document]],
    pairs_by_split: dict[str, list[tuple[int, int]]],
    features_by_split: dict[str, np.ndarray],
    targets_by_split: dict[str, np.ndarray],
    rows_by_split: dict[str, list[dict[str, object]]],
    vocabulary: list[str],
    word_index: dict[str, int],
    category_names: list[str],
    requested_pairs: dict[str, int],
) -> list[str]:
    """Check the claims the rest of the project relies on. Raise if any fail."""

    checks: list[str] = []
    vocabulary_size = len(vocabulary)

    id_sets = {split: {document.doc_id for document in documents} for split, documents in splits.items()}
    if len(id_sets["train"] & id_sets["val"]) != 0:
        raise RuntimeError("A document id appears in both the training and validation splits.")
    if len(id_sets["train"] & id_sets["test"]) != 0:
        raise RuntimeError("A document id appears in both the training and test splits.")
    if len(id_sets["val"] & id_sets["test"]) != 0:
        raise RuntimeError("A document id appears in both the validation and test splits.")
    checks.append("document ids are disjoint across train, validation, and test")

    origins = {
        split: {document.origin for document in documents}
        for split, documents in splits.items()
    }
    if origins["train"] != {"official_train"} or origins["val"] != {"official_train"}:
        raise RuntimeError("Train and validation documents must come from the official training subset.")
    if origins["test"] != {"official_test"}:
        raise RuntimeError("Test documents must come from the official test subset.")
    checks.append("validation was cut from the official training posts, and test was left intact")

    if set(category_names) != EXPECTED_CATEGORIES:
        raise RuntimeError(f"Expected the 20 Newsgroups categories. Found {category_names!r}.")
    for split, documents in splits.items():
        found = {document.category for document in documents}
        missing = [name for name in category_names if name not in found]
        if missing:
            raise RuntimeError(f"The {split} document split is missing categories: {missing}.")
    checks.append("all 20 categories are present in the train, validation, and test documents")

    for split, rows in rows_by_split.items():
        features = features_by_split[split]
        targets = targets_by_split[split]
        documents = splits[split]
        pairs = pairs_by_split[split]
        if len(rows) != requested_pairs[split]:
            raise RuntimeError(
                f"The {split} split has {len(rows)} pairs, not the requested {requested_pairs[split]}."
            )
        if features.shape != (requested_pairs[split], vocabulary_size * 2):
            raise RuntimeError(
                f"The {split} feature matrix has shape {features.shape}, expected "
                f"{(requested_pairs[split], vocabulary_size * 2)}."
            )
        if targets.shape != (requested_pairs[split],):
            raise RuntimeError(f"The {split} target vector has shape {targets.shape}.")
        if not np.isin(features, (0, 1)).all():
            raise RuntimeError(f"The {split} features contain values other than 0 and 1.")
        if int((targets == 1).sum()) != int((targets == 0).sum()):
            raise RuntimeError(f"The {split} pairs are not balanced between Y=1 and Y=0.")

        seen: set[tuple[str, str]] = set()
        documents_by_id = {document.doc_id: document for document in documents}
        for row_index, row in enumerate(rows):
            post_a = documents_by_id[str(row["doc_a_id"])]
            post_b = documents_by_id[str(row["doc_b_id"])]
            key = _unordered_key(post_a.doc_id, post_b.doc_id)
            if key in seen:
                raise RuntimeError(f"Duplicate or reversed pair in {split}: {key}.")
            seen.add(key)
            expected_y = 1 if post_a.category == post_b.category else 0
            if int(row["y"]) != expected_y or int(targets[row_index]) != expected_y:
                raise RuntimeError(
                    f"Pair {row['pair_id']} has Y={row['y']}, but the categories are "
                    f"{post_a.category!r} and {post_b.category!r}."
                )
            if pairs[row_index][0] == pairs[row_index][1]:
                raise RuntimeError(f"Pair {row['pair_id']} uses the same post twice.")
            recomputed = recompute_feature_row(post_a, post_b, word_index, vocabulary_size)
            if not np.array_equal(features[row_index], recomputed):
                raise RuntimeError(f"Recomputed features do not match pair {row['pair_id']}.")
            active_a = int(recomputed[:vocabulary_size].sum())
            active_b = int(recomputed[vocabulary_size:].sum())
            if active_a != int(row["n_features_a"]) or active_b != int(row["n_features_b"]):
                raise RuntimeError(f"Active-feature counts do not match pair {row['pair_id']}.")
    checks.append("pair counts, balance, uniqueness, labels, and feature rows match")
    checks.append("every feature row was recomputed from the two posts and matched")
    return checks


def format_pair_preview(
    row: dict[str, object],
    documents_by_id: dict[str, Document],
    features: np.ndarray,
    row_index: int,
    vocabulary: list[str],
) -> str:
    post_a = documents_by_id[str(row["doc_a_id"])]
    post_b = documents_by_id[str(row["doc_b_id"])]
    vocabulary_size = len(vocabulary)
    words_a = words_in_block(features[row_index, :vocabulary_size], vocabulary)
    words_b = words_in_block(features[row_index, vocabulary_size:], vocabulary)
    shared = sorted(set(words_a) & set(words_b))
    return "\n".join(
        [
            f"Pair {row['pair_id']}",
            f"  Y = {row['y']}  ({'same category' if row['y'] == 1 else 'different categories'})",
            f"  Post A: {post_a.doc_id}",
            f"    category: {post_a.category}",
            f"    text: {preview_text(post_a.text)}",
            f"    most common active words: {', '.join(words_a[:8]) if words_a else '(none)'}",
            f"    least common active words: {', '.join(words_a[-8:]) if words_a else '(none)'}",
            f"  Post B: {post_b.doc_id}",
            f"    category: {post_b.category}",
            f"    text: {preview_text(post_b.text)}",
            f"    most common active words: {', '.join(words_b[:8]) if words_b else '(none)'}",
            f"    least common active words: {', '.join(words_b[-8:]) if words_b else '(none)'}",
            f"  shared vocabulary words: {', '.join(shared[:12]) if shared else '(none)'}",
        ]
    )


def select_preview_rows(rows: list[dict[str, object]], per_class: int) -> list[dict[str, object]]:
    selected = []
    for target in (1, 0):
        found = [row for row in rows if int(row["y"]) == target][:per_class]
        if len(found) < per_class:
            raise RuntimeError(f"Could not find {per_class} preview pairs with Y={target}.")
        selected.extend(found)
    return selected


def output_directory(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    fraction = str(args.val_fraction).replace(".", "p")
    name = (
        f"s{args.seed}_f{args.max_features}_t{args.train_pairs}"
        f"_v{args.val_pairs}_e{args.test_pairs}_val{fraction}"
    )
    return ROOT / "prepared" / name


def save_prepared_data(
    output_dir: Path,
    settings: dict,
    vocabulary: list[str],
    document_frequency: list[int],
    inventory: list[dict[str, object]],
    rows_by_split: dict[str, list[dict[str, object]]],
    features_by_split: dict[str, np.ndarray],
    targets_by_split: dict[str, np.ndarray],
    preview: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output_dir / "vocabulary.json").write_text(
        json.dumps(
            {
                "words": vocabulary,
                "document_frequency": document_frequency,
                "selection": (
                    "highest document frequency on training posts only; "
                    "ties broken alphabetically"
                ),
                "token_pattern": TOKEN_PATTERN.pattern,
                "lowercase": True,
                "minimum_token_length": 2,
                "stopwords_removed": False,
                "stemming": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(output_dir / "documents.csv", inventory)
    for split, rows in rows_by_split.items():
        write_csv(output_dir / f"pairs_{split}.csv", rows)
        np.savez_compressed(
            output_dir / f"features_{split}.npz",
            X=features_by_split[split],
            y=targets_by_split[split],
        )
    (output_dir / "preview.txt").write_text(preview, encoding="utf-8")


def self_check() -> list[str]:
    """Check the pair and feature logic on a hand-written example.

    This example is illustrative. It is not a 20 Newsgroups pair.
    """

    sports = [
        Document(f"sports-{index}", "sports", "official_train", "hockey", frozenset({"hockey", "team"}))
        for index in range(3)
    ]
    space = [
        Document(f"space-{index}", "space", "official_train", "orbit", frozenset({"orbit", "team"}))
        for index in range(3)
    ]
    documents = sports + space
    pairs = sample_balanced_pairs(documents, n_pairs=4, seed=7, split="illustrative")
    keys = []
    for left, right in pairs:
        if left == right:
            raise RuntimeError("Illustrative self-check produced a self-pair.")
        keys.append(_unordered_key(documents[left].doc_id, documents[right].doc_id))
    if len(keys) != len(set(keys)):
        raise RuntimeError("Illustrative self-check produced a duplicate or reversed pair.")

    vocabulary = ["hockey", "team", "orbit"]
    word_index = {word: index for index, word in enumerate(vocabulary)}
    features, targets = build_feature_matrix([(0, 3)], documents, word_index, len(vocabulary))
    expected = np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8)
    if not np.array_equal(features[0], expected) or int(targets[0]) != 0:
        raise RuntimeError(
            "Illustrative feature-block check failed. "
            f"Got features {features[0].tolist()} and Y={int(targets[0])}."
        )

    try:
        sample_balanced_pairs(documents[:2], n_pairs=4, seed=1, split="too-small")
    except RuntimeError as error:
        if "Impossible pair request" not in str(error):
            raise
    else:
        raise RuntimeError("A too-large pair request did not raise an error.")

    return [
        "illustrative pair sampler rejected self-pairs and reversed duplicates",
        "illustrative feature blocks put Post A before Post B",
        "an impossible pair count raises a clear error",
    ]


def build_settings(
    args: argparse.Namespace,
    output_dir: Path,
    category_names: list[str],
    splits: dict[str, list[Document]],
    rows_by_split: dict[str, list[dict[str, object]]],
    vocabulary: list[str],
    archive_info: dict[str, str],
) -> dict:
    empty_counts = {
        split: sum(1 for document in documents if len(document.tokens) == 0)
        for split, documents in splits.items()
    }
    return {
        "project": "pairrot",
        "stage": "prepared-pairs",
        "output_dir": str(output_dir),
        "assignment_requirements": ASSIGNMENT_REQUIREMENTS,
        "experiment_choices": {
            "validation_fraction": args.val_fraction,
            "validation_split": "stratified by category, using scikit-learn train_test_split",
            "max_features": args.max_features,
            "vocabulary_ranking": "document frequency on training posts, alphabetical tie break",
            "tokenization": "lowercase alphabetic tokens with length at least 2",
            "stopwords_removed": False,
            "stemming": False,
            "empty_posts": "kept",
            "pair_balance": "exactly half Y=1 and half Y=0",
            "pair_sampling": "uniform among unordered pairs inside each label class",
            "pair_orientation": "random assignment of the two posts to Post A and Post B",
            "seed": args.seed,
            "document_split_seed": args.seed,
            "pair_seeds": {split: args.seed + offset for split, offset in PAIR_SEED_OFFSET.items()},
            "train_pairs": args.train_pairs,
            "val_pairs": args.val_pairs,
            "test_pairs": args.test_pairs,
        },
        "dependencies": {
            "python": _python_version(),
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
        },
        "dataset_archive": archive_info,
        "measured": {
            "categories": category_names,
            "documents_per_split": {split: len(documents) for split, documents in splits.items()},
            "category_counts": {
                split: category_counts(documents) for split, documents in splits.items()
            },
            "empty_documents_after_tokenization": empty_counts,
            "vocabulary_size": len(vocabulary),
            "feature_width": len(vocabulary) * 2,
            "pairs_per_split": {split: len(rows) for split, rows in rows_by_split.items()},
            "pair_list_sha256": {split: digest_pairs(rows) for split, rows in rows_by_split.items()},
        },
    }


def _python_version() -> str:
    import sys

    return sys.version.split()[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Pairrot's 20 Newsgroups pairs.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--train-pairs", type=int, default=1000)
    parser.add_argument("--val-pairs", type=int, default=200)
    parser.add_argument("--test-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=457)
    parser.add_argument("--preview-per-class", type=int, default=2)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--data-home",
        type=str,
        default=str(ROOT / "cache"),
        help="Directory for the scikit-learn 20 Newsgroups download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Illustrative self-checks, not 20 Newsgroups examples:")
    for check in self_check():
        print(f"  PASS {check}")

    data_home = Path(args.data_home)
    data_home.mkdir(parents=True, exist_ok=True)
    archive_info = ensure_newsgroups_archive(data_home)
    print(f"Dataset archive: {archive_info}")
    print("Loading the official 20 Newsgroups training and test posts...")
    official_train, train_categories = load_official_subset("train", data_home)
    official_test, test_categories = load_official_subset("test", data_home)
    if train_categories != test_categories:
        raise RuntimeError("The official train and test subsets do not use the same categories.")
    category_names = train_categories
    if len(category_names) != 20:
        raise RuntimeError(f"Expected all 20 categories. The loader returned {len(category_names)}.")

    train_ids = [document.doc_id for document in official_train]
    test_ids = [document.doc_id for document in official_test]
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise RuntimeError("A document id is repeated inside an official subset.")
    if set(train_ids) & set(test_ids):
        raise RuntimeError("An official training document id also appears in the official test subset.")

    train_documents, val_documents = split_training_documents(
        official_train,
        args.val_fraction,
        args.seed,
    )
    splits = {"train": train_documents, "val": val_documents, "test": list(official_test)}
    split_train_ids = {document.doc_id for document in train_documents}
    split_val_ids = {document.doc_id for document in val_documents}
    if split_train_ids | split_val_ids != set(train_ids) or split_train_ids & split_val_ids:
        raise RuntimeError("The validation split does not partition the official training documents.")
    print(
        "Document counts after the validation split: "
        + ", ".join(f"{split}={len(documents)}" for split, documents in splits.items())
    )

    vocabulary, document_frequency = select_vocabulary(train_documents, args.max_features)
    word_index = {word: index for index, word in enumerate(vocabulary)}
    print("First 15 training vocabulary words, with document frequency:")
    for word, count in list(zip(vocabulary, document_frequency))[:15]:
        print(f"  {word:<20} {count}")

    requested_pairs = {
        "train": args.train_pairs,
        "val": args.val_pairs,
        "test": args.test_pairs,
    }
    pairs_by_split = {}
    features_by_split = {}
    targets_by_split = {}
    rows_by_split = {}
    for split, documents in splits.items():
        pairs = sample_balanced_pairs(
            documents,
            requested_pairs[split],
            args.seed + PAIR_SEED_OFFSET[split],
            split,
        )
        features, targets = build_feature_matrix(pairs, documents, word_index, len(vocabulary))
        rows = pair_rows_for_split(split, pairs, documents, features, targets, len(vocabulary))
        pairs_by_split[split] = pairs
        features_by_split[split] = features
        targets_by_split[split] = targets
        rows_by_split[split] = rows
        print(
            f"  {split}: {len(rows)} pairs, shape {features.shape}, "
            f"Y=1 {(targets == 1).sum()}, Y=0 {(targets == 0).sum()}"
        )

    checks = verify_prepared_data(
        splits,
        pairs_by_split,
        features_by_split,
        targets_by_split,
        rows_by_split,
        vocabulary,
        word_index,
        category_names,
        requested_pairs,
    )
    print("Verification:")
    for check in checks:
        print(f"  PASS {check}")

    preview_sections = [
        "Real 20 Newsgroups pairs. Categories are shown for checking; they are not model features.",
        "",
    ]
    for split, rows in rows_by_split.items():
        documents_by_id = {document.doc_id: document for document in splits[split]}
        preview_sections.append(f"## {split}")
        for row in select_preview_rows(rows, args.preview_per_class):
            row_index = int(str(row["pair_id"]).split("-")[1])
            preview_sections.append(
                format_pair_preview(
                    row,
                    documents_by_id,
                    features_by_split[split],
                    row_index,
                    vocabulary,
                )
            )
            preview_sections.append("")
    preview = "\n".join(preview_sections)

    output_dir = output_directory(args)
    settings = build_settings(
        args,
        output_dir,
        category_names,
        splits,
        rows_by_split,
        vocabulary,
        archive_info,
    )
    inventory = []
    for split, documents in splits.items():
        inventory.extend(document_inventory(split, documents))
    save_prepared_data(
        output_dir,
        settings,
        vocabulary,
        document_frequency,
        inventory,
        rows_by_split,
        features_by_split,
        targets_by_split,
        preview,
    )
    print(preview)
    print(f"Saved prepared pairs to {output_dir}")


if __name__ == "__main__":
    main()
