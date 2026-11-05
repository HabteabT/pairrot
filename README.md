# Pairrot

Pairrot is an IKT457 project that compares vanilla and hierarchical Tsetlin
Machines on whether two posts belong to the same newsgroup category. The
first stage prepares the shared input data.

## Prepare the data

Use Python 3.11 or later. In a virtual environment, run:

```sh
python -m pip install -r requirements.txt
python prepare_data.py
```

The first run downloads 20 Newsgroups. Use `python prepare_data.py --help` to
change pair counts, seed, vocabulary size, or paths. Pair counts must be even.
For a small run, use:

```sh
python prepare_data.py --train-pairs 100 --validation-pairs 40 --test-pairs 40 --output-dir artifacts/pilot
```

## Read the output

The default output is `artifacts/data/`. The files for each split share the
same row order.

| File | Contents |
| --- | --- |
| `*_features.npz` | SciPy sparse binary matrix with the Post A vocabulary block followed by the Post B block |
| `*_targets.npy` | One target per pair, with 1 for the same category and 0 for different categories |
| `*_pairs.csv` | Post IDs, category names, and targets for checking the pairs |
| `metadata.json` | Ordered vocabulary, split counts, seed, cleaning settings, and library versions |

Use the feature matrices as model inputs and the target arrays as labels.
The CSV category columns are for inspection. For a vocabulary of 1,000 words,
each pair has 2,000 features.

The script keeps the official test posts separate. It holds out 20% of the
official training posts for validation and learns the vocabulary from the
remaining posts. It samples uniformly within each target class, removes
duplicate unordered pairs by construction, and randomizes which post appears
first. Use validation for tuning and the test pairs for the final comparison.
