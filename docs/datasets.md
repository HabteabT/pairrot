# Datasets and what the models use

This note lists every dataset in the project, what each prepared folder contains, and which folder the saved models were trained on. The official test pairs exist in every folder. They have not been scored.

## Source

The posts are the by-date 20 Newsgroups collection used by scikit-learn, all 20 categories. The local archive is `dataset/cache/20news-bydate_py3.pkz`. Its checksum is `8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610`.

The official split is about 18,846 posts: 11,314 training posts and 7,532 test posts. The assignment figure of 20,000 is the usual round number for this collection. This project does not have a separate file of 20,000 posts.

A validation set is cut from the official training posts. It is 20 percent, stratified by category, seed 457. That split is the same in every prepared folder:

| Split | Posts | Where the posts come from |
| --- | ---: | --- |
| Train | 9,051 | Official training posts, after the validation cut |
| Validation | 2,263 | Official training posts, held out from training |
| Test | 7,532 | Official test posts |

The category counts for that split are stored in each folder's `settings.json`. The smallest training category is `talk.religion.misc` (302 posts). The largest is `rec.sport.hockey` (480 posts).

## How a pair is built

Each example is two distinct posts from the same split. Y = 1 when they share a category. Y = 0 when they do not. Each split is exactly half Y = 1 and half Y = 0. The two posts are assigned to Post A and Post B at random. Pair seeds are 458 for train, 459 for validation, and 460 for test.

Tokens are lowercase letters of length at least 2. There is no stemming. Empty posts are kept.

One vocabulary is fit on the training posts only. A feature is 1 when the word is present and 0 when it is absent. The model row is Post A, then Post B, so the width is twice the vocabulary size. Category names are used to build Y and, in one folder, to choose the vocabulary. They are not columns in `X`.

The model files are the `.npz` files. Each one contains only `X` and `y`. The CSV files are for reading the pairs. They are not model inputs.

| File | What it is |
| --- | --- |
| `settings.json` | Counts, seeds, vocabulary rule, and the measured shapes |
| `vocabulary.json` | The shared word list, in column order |
| `documents.csv` | One row per post, with its split and category |
| `pairs_train.csv`, `pairs_val.csv`, `pairs_test.csv` | Pair id, the two posts, the two categories, and Y |
| `features_train.npz`, `features_val.npz`, `features_test.npz` | `X` and `y` only |
| `preview.txt` | A few pairs with the post text |

The folder name records the settings:

`s{seed}_f{words}_t{train pairs}_v{validation pairs}_e{test pairs}_val{fraction}`

`_fulltext` means headers, footers, and quotes were kept. `_topical` means the vocabulary was chosen by chi-square against the training categories. A name without those suffixes is the cleaned frequency vocabulary: headers, footers, and quotes removed, words ranked by how many training posts contain them.

## The four prepared folders

All four are under `dataset/prepared/`.

### The folder the saved models use

`s457_f1000_t10000_v2000_e2000_val0p2`

| Item | Value |
| --- | --- |
| Posts | Headers, footers, and quotes removed |
| Vocabulary | 1,000 most frequent words. Stop words are kept. Categories do not choose the words. |
| First words | the, to, and, of, in, is, it, that, for, this, have, you |
| Pairs | 10,000 train, 2,000 validation, 2,000 test |
| Feature shape | `(pairs, 2000)` |
| Used by | The saved vanilla model and the saved hierarchical model |

Both models use s = 1.0 with the boost flag off. A true literal is not included. That is the reasoning-by-elimination rule.

### An earlier experiment, not the saved model

`s457_f300_t30000_v2000_e2000_val0p2_fulltext_topical`

| Item | Value |
| --- | --- |
| Posts | Full posts. Headers, footers, and quotes are kept. |
| Vocabulary | 300 words, topical. Chi-square of word presence against the category, on training posts only. |
| First words | clipper, encryption, sale, bike, dod, hockey, israel, windows, israeli, god, nhl, gun |
| Pairs | 30,000 train, 2,000 validation, 2,000 test |
| Feature shape | `(pairs, 600)` |
| Empty posts after tokenization | 0 / 0 / 0 |
| Used by | Nothing on the dashboard. The vanilla state from that run is archived at `code/results/archive/20261117-212242_val0.711`. |

That run kept the headers, chose the words with the training categories, and turned boost on. Those three choices are outside the required run.

### The required cleaned folder

`s457_f1000_t1000_v200_e200_val0p2`

| Item | Value |
| --- | --- |
| Posts | Headers, footers, and quotes removed |
| Vocabulary | 1,000 most frequent words. Stop words are kept. |
| First words | the, to, and, of, in, is, it, that, for, this, have, you |
| Pairs | 1,000 train, 200 validation, 200 test |
| Feature shape | `(pairs, 2000)` |
| Empty posts after tokenization | 255 train, 56 validation, 226 test |
| Used by | `train.bat`, the required cleaned vanilla run. It is not the model on the dashboard. |

The early vanilla runs on this folder stayed near 0.50. The saved result from that setup is validation accuracy 0.505, with the model predicting "different category" for almost every pair. That result is archived and is not the dashboard score.

### Two unused vocabulary trials

These two folders were built and not used for a saved model.

`s457_f1000_t4000_v400_e200_val0p2_stop`

| Item | Value |
| --- | --- |
| Posts | Cleaned |
| Vocabulary | 1,000 frequent words after English stop words are removed |
| First words | like, just, don, know, think, does, people, time, use, good, new, make |
| Pairs | 4,000 train, 400 validation, 200 test |
| Empty posts | 262 train, 57 validation, 235 test |

`s457_f1000_t4000_v400_e200_val0p2_stop_disc`

| Item | Value |
| --- | --- |
| Posts | Cleaned. The pair lists match the `_stop` folder. |
| Vocabulary | 1,000 words ranked by category concentration on the training posts, minimum 20 documents. The current `prepare_data.py` no longer has this ranking. The folder is the record of that trial. |
| First words | windows, god, space, car, sale, key, bike, team, encryption, gun, israel, jesus |
| Pairs | 4,000 train, 400 validation, 200 test |
| Empty posts | 262 train, 57 validation, 235 test |

## What the saved models use

Both saved models load the same two files from the cleaned 10,000-pair folder:

- `features_train.npz`
- `features_val.npz`

They do not load `features_test.npz`.

The same pairs were trained twice. The only change is the boost flag. s stays 1.0. Test pairs were not read.

| Setting | Model | Training accuracy | Validation accuracy | Confusion matrix |
| --- | --- | ---: | ---: | --- |
| Boost off | Vanilla, epoch 2 of 30 | 0.511 | 0.517 | `[[274, 726], [239, 761]]` |
| Boost off | Hierarchical, epoch 10 of 24 | 0.500 | 0.501 | `[[1000, 0], [998, 2]]` |
| Boost on | Vanilla, epoch 8 of 12 | 0.720 | 0.521 | `[[534, 466], [491, 509]]` |
| Boost on | Hierarchical, epoch 9 of 12 | 0.566 | 0.529 | `[[775, 225], [717, 283]]` |

Boost off does not add a present word to a clause. Both models stay next to chance. Boost on lets a firing clause include every true literal. Training accuracy rises, and validation rises only a little: vanilla 0.517 to 0.521, hierarchical 0.501 to 0.529.

The dashboard currently shows the boost-on run, 52.1% and 52.9%. The boost-off vanilla model is in `code/results/archive/20261117-224250_val0.517`. The boost-off hierarchical model is in `hierarchical/results/archive/boost0_val0.501`. The hierarchical tree is a Python training loop, not the CUDA library, because this computer has no NVIDIA GPU.

## Numbers that are not this computer's saved models

The experiment log records other runs. They are not the files the dashboard loads.

| Number | What it is |
| --- | --- |
| 0.505 | Cleaned folder, boost off, the required-style vanilla run on this machine |
| 0.692 | Epoch 8 of the 300-word run on this machine, before training continued. The saved state is the later epoch, 0.711. |
| 0.759 | The same settings on another machine. It is not a result of this computer. |
| Test accuracy | Not measured. The test pairs stay unread until the final comparison. |
