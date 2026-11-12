# Dataset

This folder only builds the pairs. It does not train a model, and it does not serve a page.

## What it does

1. Load the official 20 Newsgroups training posts and test posts.
2. Cut a stratified validation set from the training posts.
3. Fit one vocabulary on the training posts.
4. Sample balanced pairs inside each split.
5. Save `X` and `y` for the model, plus a CSV so a person can check the categories.

## Command

From the repository root:

```powershell
.\.venv\Scripts\python -m pip install -r dataset\requirements.txt
.\.venv\Scripts\python dataset\prepare_data.py
```

The prepared files from the first verified run are in:

`dataset/prepared/s457_f1000_t1000_v200_e200_val0p2`

| File | Role |
| --- | --- |
| `settings.json` | Counts, seeds, and the choices used for this run. |
| `vocabulary.json` | The shared 1,000-word list. |
| `documents.csv` | One row per post and which split it belongs to. |
| `pairs_train.csv`, `pairs_val.csv`, `pairs_test.csv` | Document ids, categories, and Y. For inspection. |
| `features_train.npz`, `features_val.npz`, `features_test.npz` | `X` and `y` only. This is what the model code loads. |
| `preview.txt` | A few real pairs with text. |

`dataset/cache/` is the downloaded archive. It is not committed.

## Checks that passed on the first run

- Train 9,051 posts, validation 2,263 posts, test 7,532 posts.
- No document id is in more than one split.
- 1,000 / 200 / 200 pairs, each split half Y=1 and half Y=0.
- Feature matrices have shape `(pairs, 2000)`.
- Every feature row was recomputed and matched.

## Options

| Option | What it does |
| --- | --- |
| `--full-text` | Keeps headers, footers, and quoted text. The default removes them. The course brief does not require the removal. |
| `--vocabulary topical` | Chooses the words that best separate the categories on the training posts (chi-square), without stop words. The default, `frequency`, takes the most common words. |
| `--train-pairs`, `--val-pairs`, `--test-pairs` | Pair counts. 200 validation pairs cannot separate 0.52 from 0.50, so 2,000 is a better size. |

Each option changes the output folder name, so runs do not overwrite each other.
