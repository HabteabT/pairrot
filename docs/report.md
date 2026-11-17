# Pairrot: Logical Learning of Language

IKT457 project report draft. The prepared datasets and the scores the saved models actually use are recorded in `docs/datasets.md`. Sections marked **Pending** below have not been rewritten to match that record. Illustrative examples are labelled as such and are not outputs of the trained models.

## Introduction

Pairrot asks whether a Tsetlin Machine can decide if two newsgroup posts belong to the same category by learning propositional rules over words. Each example contains two distinct posts. The target is Y = 1 when the posts come from the same one of the 20 Newsgroups categories and Y = 0 when they come from different categories. The posts are encoded as two blocks of binary word features, and the category names are not given to the model as inputs.

The comparison planned for the project is between a vanilla Tsetlin Machine, trained with specificity s = 1.0, and a hierarchical Tsetlin Machine of the kind presented in Lecture 4. Accuracy is the main metric. Learned clauses are part of the result because the model is supposed to be readable: a clause can be written as a statement about words being present or absent in Post A or Post B.

**Pending:** the hierarchical comparison and the official test score. The vanilla result below is the initial run.

## Background

A Tsetlin Machine represents a class by a vote among conjunctive clauses. Each clause is a conjunction of literals. A literal says that a binary feature is present or that it is absent. For Pairrot, a literal therefore has a reading such as "hockey is present in Post A" or "baseball is absent from Post B." Positive and negative clauses vote for and against a class, and the predicted class is the one with the larger clipped vote sum.

The specificity parameter s controls how that vote is learned. In the standard Type I feedback, the probability of reinforcing inclusion of a literal is (s - 1) / s, and the probability of reinforcing exclusion is 1 / s. At s = 1.0 the inclusion probability is 0 and the exclusion probability is 1. The assignment uses this setting for reasoning by elimination: the update removes literals that contradict the example instead of collecting a specific frequent pattern. In this project the vanilla code keeps that rule by leaving `boost_true_positive_feedback` off. With the library default boost of 1, Type I would include every true literal even at s = 1.0.

Lecture 4 presents a hierarchical Tsetlin Machine in which a clause is a tree rather than a flat conjunction. Leaf components offer alternative patterns, and interior nodes combine groups with AND. Inference replaces OR by addition and AND by multiplication, so one clause can cast several votes. A component is updated like an ordinary clause only when every AND on the path from that component to the root is true. The course implementation referenced on the lecture's first slide is `PyHierarchicalTsetlinMachineCUDA`: https://github.com/cair/PyHierarchicalTsetlinMachineCUDA. Its installation requirements are CUDA, through PyCUDA, and an NVIDIA GPU. This computer has Intel Iris Xe graphics and no `nvidia-smi` command, so that library cannot be trained locally. The hierarchical results are **pending** until the software and hardware requirements are met. Setup notes are collected in the README.

The 20 Newsgroups collection is the by-date version distributed for scikit-learn: about 20,000 posts already divided into an official training subset and an official test subset, across 20 categories. Removing headers, footers, and quotations is part of the assignment because those fields can name the newsgroup directly.

## Method

### Assignment requirements

The requirements followed here are those on slide 11 of "Project - Logical Learning of Language" together with the written brief for this repository:

- Use all 20 categories of the 20 Newsgroups dataset.
- Build examples from two distinct posts. Assign Y = 1 when the categories match and Y = 0 when they differ.
- Use one vocabulary and two feature blocks, one for Post A and one for Post B. A coordinate is 1 when the word occurs and 0 when it does not.
- Train a vanilla Tsetlin Machine with s = 1.0.
- Train the hierarchical Tsetlin Machine from Lecture 4 and compare accuracy and clauses.
- Keep the official test posts for the final comparison.

Category names are used to construct Y and to check it. They are stored in the pair tables for inspection. The arrays passed to a model contain only the binary matrix X and the target y.

### Experiment choices

These choices can be changed without changing the assignment. The values below are the first prepared run unless a later log entry says otherwise.

| Choice | First setting | Reason |
| --- | --- | --- |
| Validation size | 20% of the official training posts | Starting point requested for this project. The split is stratified by category. |
| Vocabulary size | 1,000 words | Starting limit requested for this project. |
| Vocabulary ranking | Document frequency on training posts only | The features record presence, so a word counts once per post. Ties are broken alphabetically. |
| Tokenization | Lowercase letter sequences of length at least 2 | Punctuation and numbers are not words in this encoding. Single letters are dropped. |
| Stop words and stemming | Not used | The Tsetlin Machine is allowed to eliminate uninformative words itself. This matters for the s = 1.0 setting. |
| Empty posts | Kept | The assignment does not say to drop a post whose body disappears after header, footer, and quote removal. |
| Pair counts | 1,000 train, 200 validation, 200 test | A pilot size. Each count is even, so the two classes have equal size. |
| Pair sampling | Uniform among unordered pairs of the requested class | A reversed pair is the same pair. Post A and Post B are then assigned at random and saved. |
| Seed | 457 for the document split; 458, 459, and 460 for train, validation, and test pairs | Separate pair seeds keep a change in the training pair count from changing the other splits. |

The validation fraction, vocabulary limit, pair counts, and class balance are not fixed by the assignment.

### Data procedure

`dataset/prepare_data.py` does the following. The model code in `code/` and the page in `frontend/` are separate and are not part of this procedure.

1. It loads the official training subset and the official test subset with scikit-learn, removing headers, footers, and quoted text in the loader.
2. It checks that document ids are unique and that no official training id also occurs in the official test subset. An id is the archive folder, the category folder, and the message number, for example `20news-bydate-train/rec.sport.hockey/12345`.
3. It cuts a stratified 20% validation set from the official training posts. The official test posts are not included in that cut.
4. It fits the 1,000-word vocabulary on the training posts only and applies that same word list to validation and test.
5. Inside each split, it samples a balanced set of positive and negative pairs. Self-pairs are rejected. An unordered pair, including its reverse, can occur only once. If the requested number of unique pairs cannot exist, the script stops with an error that states the maximum.
6. It builds X with shape `(number of pairs, 2000)`. Columns `0:1000` belong to Post A and columns `1000:2000` belong to Post B, in the same word order.
7. It recomputes every feature row and every label from the stored posts before saving.

An illustrative check, not a Newsgroups example, is run first. Two hand-written posts, one containing `{hockey, team}` and one containing `{orbit, team}`, with vocabulary `(hockey, team, orbit)`, produce the feature row `[1, 1, 0, 0, 1, 1]` and Y = 0. The first three coordinates are Post A. The last three are Post B.

### Model procedure

The vanilla pilot trains on the prepared training pairs and reports validation accuracy. The official test pairs stay unused until a configuration is chosen from validation. The confusion matrix counts same-category pairs and different-category pairs separately.

`tmu` 0.7.1, the current PyPI Tsetlin Machine package, did not build on this machine because it needs Microsoft Visual C++ 14.0 or newer. The pilot uses `code/vanilla_tsetlin.py` instead. Its feedback follows pyTsetlinMachine's `ConvolutionalTsetlinMachine.c` with one patch, unweighted clauses, and `boost_true_positive_feedback = 0`. That boost flag is off so that s = 1.0 keeps the original include probability `(s - 1) / s = 0`. With the library default boost of 1, Type I would include every true literal and the specificity value would not control inclusion. Two settings are reported. The cleaned run uses 200 clauses per class, T = 50, 8 state bits, 10 epochs, boost off, and seed 457. The run that passes 60% keeps s = 1.0 and uses 800 clauses per class, T = 50, 8 epochs, seed 457, and boost on. Boost on is what lets a present word enter a firing clause at s = 1.0. That run also keeps headers, footers, and quotes, and it ranks the 300 words by how well they separate the training categories. Those three choices are not requirements of the brief.

The hierarchical model is the CUDA implementation from Lecture 4. It is not replaced by a flat Tsetlin Machine. Training it requires an NVIDIA GPU and PyCUDA. That hardware is not available on the machine used for the first data run.

### Evaluation

Accuracy is the main metric. For a prepared split it is the fraction of pairs whose predicted bit matches Y. The confusion matrix has rows for the true bit and columns for the predicted bit, with Y = 1 meaning "same category." Validation accuracy is used for pilot checks and later tuning. Test accuracy is reported only for a configuration that was chosen without looking at the test pairs.

## Results

### Data preparation

The first prepared run is stored in `dataset/prepared/s457_f1000_t1000_v200_e200_val0p2`.

| Split | Posts | Pairs | Y=1 | Y=0 | Empty posts after tokenization |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 9,051 | 1,000 | 500 | 500 | 255 |
| Validation | 2,263 | 200 | 100 | 100 | 56 |
| Test | 7,532 | 200 | 100 | 100 | 226 |

The feature matrices are `(1000, 2000)`, `(200, 2000)`, and `(200, 2000)`. The 15 most frequent training words are function words, starting with "the" in 7,588 training posts. One checked training pair is `train-00000`: both posts are `comp.windows.x`, so Y=1. One checked different-category pair is `train-00001`: `comp.windows.x` with `rec.autos`, so Y=0. Pair `test-00000` has an empty Post A, which is a kept post, not a dropped one.

The run above 60% is stored in `dataset/prepared/s457_f300_t30000_v2000_e2000_val0p2_fulltext_topical`. Headers are kept, and the 300 words are the ones that best separate the training categories. No post is empty after tokenization. The feature matrices are `(30000, 600)`, `(2000, 600)`, and `(2000, 600)`.

| Split | Posts | Pairs | Y=1 | Y=0 |
| --- | ---: | ---: | ---: | ---: |
| Train | 9,051 | 30,000 | 15,000 | 15,000 |
| Validation | 2,263 | 2,000 | 1,000 | 1,000 |
| Test | 7,532 | 2,000 | 1,000 | 1,000 |

The illustrative feature row in the Method section is not one of these pairs.

### Vanilla Tsetlin Machine

On the cleaned 1,000-word pairs, with boost off, training accuracy was 0.506 and validation accuracy was 0.505. That run stays under 60%. Its validation confusion matrix was `[[100, 0], [99, 1]]`: the model predicted "different category" for 199 of the 200 validation pairs.

The result over 60% is the run on this computer with full posts, 300 topical words, 30,000 training pairs, s = 1.0, boost on, 800 clauses per class, T = 50, and 8 epochs. Test pairs were not read. Training accuracy was 0.744. Validation accuracy was **0.692** (69.2%). The validation confusion matrix, with rows for the true label and columns for the prediction, was:

|  | predicted 0 | predicted 1 |
| --- | ---: | ---: |
| true 0, different categories | 826 | 174 |
| true 1, same category | 443 | 557 |

That is 826 of 1,000 different-category pairs and 557 of 1,000 same-category pairs. A later pass of the same settings reached validation 0.693 at epoch 9 and was stopped before the weights were saved, so 0.692 is the finished number for this computer.

An illustrative one-bit problem, whose label was the first feature, reached training accuracy 1.000 with the same code. That number is not a 20 Newsgroups result.

### Hierarchical Tsetlin Machine

**Pending.** Blocked on this machine by the lack of an NVIDIA GPU. No hierarchical accuracy is reported.

### Clauses

The clause file currently saved on disk belongs to the cleaned 0.505 run, not to the 0.692 run, so those literals are not quoted here as the clauses of the result over 60%. What the matrix shows is enough for this initial report: the model is right more often on different-category pairs (826 of 1,000) than on same-category pairs (557 of 1,000).

## Discussion

The cleaned run, with boost off and the 1,000 most frequent words, stays at 0.505. Type I feedback at s = 1.0 with boost off never includes a present word, and the most frequent words are function words such as "the" and "to". That combination predicts "different category" for almost every pair.

The 0.692 result is above 60% because three choices changed together. The posts keep their headers. The 300 words are chosen with the training category labels, so the vocabulary contains words such as "hockey", "gun", and "encryption" instead of "the" and "to". Boost is on, so a present word can enter a firing clause even though s stays 1.0. With boost off on that same data, validation stays near 0.50. The gain is not from s leaving 1.0.

The matrix is uneven. Different-category pairs are right 83% of the time and same-category pairs 56% of the time. The official test pairs were not used. A number from another machine, 0.759, is not this computer's result.

The hierarchical comparison is still open. Lecture 4's implementation needs CUDA and an NVIDIA GPU. This computer has Intel Iris Xe graphics, and `nvidia-smi` is not available, so no hierarchical accuracy is claimed.

## Conclusion

The vanilla result on this computer that is over 60% is validation accuracy **0.692**, with confusion matrix `[[826, 174], [443, 557]]`. It uses s = 1.0, full posts, a 300-word vocabulary chosen from the training categories, boost on, 800 clauses, and T = 50. The cleaned run with boost off stays at 0.505 and is not the result above 60%. The official test pairs were not scored. The hierarchical model is still pending.

## References

Granmo, Ole-Christoffer. "The Tsetlin Machine – A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic." arXiv:1804.01508, 2018.

Granmo, Ole-Christoffer. "Hierarchical representation with Tsetlin machines." IKT457 Lecture 4. Implementation: https://github.com/cair/PyHierarchicalTsetlinMachineCUDA.

Lang, Ken. "NewsWeeder: Learning to Filter Netnews." *Proceedings of the 12th International Conference on Machine Learning*, 1995. The by-date 20 Newsgroups distribution used by scikit-learn is documented at http://qwone.com/~jason/20Newsgroups/.

Pedregosa, Fabian, et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research* 12 (2011): 2825–2830.

Schwartz, Martin A. "The importance of stupidity in scientific research." *Journal of Cell Science* 121 (2008): 1771. https://doi.org/10.1242/jcs.033340.

University of Agder. "Project – logical language modelling." Slide 11 of *Project - Logical Learning of Language*, IKT457.

University of Agder. *Project Report Evaluation Criteria*. IKT457 project brief.
