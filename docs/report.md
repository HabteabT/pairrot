# Pairrot: Logical Learning of Language

IKT457 project report draft. Sections marked **Pending** do not yet have a measured result. Illustrative examples are labelled as such and are not outputs of the trained models.

## Introduction

Pairrot asks whether a Tsetlin Machine can decide if two newsgroup posts belong to the same category by learning propositional rules over words. Each example contains two distinct posts. The target is Y = 1 when the posts come from the same one of the 20 Newsgroups categories and Y = 0 when they come from different categories. The posts are encoded as two blocks of binary word features, and the category names are not given to the model as inputs.

The comparison planned for the project is between a vanilla Tsetlin Machine, trained with specificity s = 1.0, and a hierarchical Tsetlin Machine of the kind presented in Lecture 4. Accuracy is the main metric. Learned clauses are part of the result because the model is supposed to be readable: a clause can be written as a statement about words being present or absent in Post A or Post B.

**Pending:** the final accuracy comparison, the clause figures, and the conclusion. The data-preparation decisions below have been implemented and are updated with the checks from the first successful run.

## Background

A Tsetlin Machine represents a class by a vote among conjunctive clauses. Each clause is a conjunction of literals. A literal says that a binary feature is present or that it is absent. For Pairrot, a literal therefore has a reading such as "hockey is present in Post A" or "baseball is absent from Post B." Positive and negative clauses vote for and against a class, and the predicted class is the one with the larger clipped vote sum.

The specificity parameter s controls how that vote is learned. In the standard Type I feedback, the probability of reinforcing inclusion of a literal is (s - 1) / s, and the probability of reinforcing exclusion is 1 / s. At s = 1.0 the inclusion probability is 0 and the exclusion probability is 1. The assignment uses this setting for reasoning by elimination: the update removes literals that contradict the example instead of collecting a specific frequent pattern. This report records how the selected library implements that setting once the pilot has inspected the library code. That inspection is **pending** until the vanilla pilot is run.

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

`tmu` 0.7.1, the current PyPI Tsetlin Machine package, did not build on this machine because it needs Microsoft Visual C++ 14.0 or newer. The pilot uses `code/vanilla_tsetlin.py` instead. Its feedback follows pyTsetlinMachine's `ConvolutionalTsetlinMachine.c` with one patch, unweighted clauses, and `boost_true_positive_feedback = 0`. That boost flag is off so that s = 1.0 keeps the original include probability `(s - 1) / s = 0`. With the library default boost of 1, Type I would include every true literal and the specificity value would not control inclusion. The pilot settings, which are not assignment requirements except for s = 1.0, are 20 clauses per class, T = 10, 8 state bits, 10 epochs, and seed 457.

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

The illustrative feature row in the Method section is not one of these pairs.

### Vanilla Tsetlin Machine

The pilot trained for 10 epochs on the 1,000 training pairs and was scored on the 200 validation pairs. Test pairs were not read. Training accuracy ended at 0.540. Validation accuracy was 0.515. The validation confusion matrix, with rows for the true label and columns for the prediction, was:

|  | predicted 0 | predicted 1 |
| --- | ---: | ---: |
| true 0, different categories | 88 | 12 |
| true 1, same category | 85 | 15 |

An illustrative one-bit problem, whose label was the first feature, reached training accuracy 1.000 with the same code. That number is not a 20 Newsgroups result.

### Hierarchical Tsetlin Machine

**Pending.** Blocked on this machine by the lack of an NVIDIA GPU. No hierarchical accuracy is reported.

### Clauses

Two automata out of 160,000 were in the Include state. Both say `these present in Post B`, in class 1 clauses 14 and 18. Both clauses have positive polarity. No included literal is negated.

On 173 validation pairs the clipped votes are 0 and 0. Those ties are predicted as class 0. The other 27 pairs all have the score pattern class 0 = 0, class 1 = 2, and every one of them has the Post B feature for "these" set. Fifteen of the 27 have true Y = 1 and 12 have true Y = 0.

Pair `val-00001` is counted as correct, but no clause fires: the true label is 0 and the tie break also says 0. Pair `val-00000` is two `comp.graphics` posts, so Y = 1, and the same tie break predicts 0. Pair `val-00005` is two `talk.religion.misc` posts. Its votes are 0 and 2, so the prediction Y = 1 is produced by the "these" clauses.

The hand-written two-bit check did include a negated literal under Type II. The trained pilot did not keep one.

## Discussion

The validation accuracy of 0.515 is only a small step above a coin flip, and the confusion matrix says why it should not be read as "the model finds same-category pairs." Of 100 pairs with Y = 1, 85 were predicted as different categories. Of 100 pairs with Y = 0, 88 were predicted as different categories. The model usually predicts Y = 0.

That pattern comes from the votes, not from a topic rule. On 173 validation pairs both classes scored 0. The implementation breaks a tie by choosing class 0. The 27 pairs predicted as Y = 1 are the pairs in which Post B contains "these," which turns on the only two included literals. Fifteen of those 27 pairs really are the same category and 12 are not, so the literal is a weak and accidental signal. It is also one of the frequent function words at the top of the vocabulary, not a category name such as hockey or orbit.

This was not the clause shape the first run was expected to leave behind. The state histogram explains it. Automata start at state 127, one step below Include. Type I feedback at s = 1.0 never moves a true literal toward Include, and it always pushes false literals toward Exclude. Type II can include every literal that is false in the current example. On a 2,000-bit vector that is a very specific clause, so it usually does not fire again. The next Type I update then treats it as a clause that output 0 and decrements its automata. After ten epochs, 43,922 automata were at state 0 and only 2 were at 128. The wide elimination step was written and then erased. What remained was a single function word.

The same implementation did learn an illustrative problem whose label was equal to the first bit, with training accuracy 1.000. The Newsgroups result is therefore not evidence that the update code never learns. It is evidence that s = 1.0, 20 clauses, and this 2,000-bit encoding did not produce stable topic clauses in ten epochs.

Two data facts will matter for the next run. First, the vocabulary's most common words are function words, because stop words were kept on purpose. Second, some posts are empty after the required quote removal, so a same-category pair can have an all-zero side, as in `test-00000`. Neither fact was "fixed" after seeing it. Both are recorded so a later comparison can say what changed.

The hierarchical comparison is still open. Lecture 4's implementation needs CUDA and an NVIDIA GPU. This computer has Intel Iris Xe graphics, and `nvidia-smi` is not available, so no hierarchical accuracy is claimed.

## Conclusion

The pair pipeline is in place: official train and test posts are separated, the vocabulary is fit on the training posts, and both models can be given the same `X` and `y`. The first vanilla pilot, with s = 1.0, reaches validation accuracy 0.515 and keeps only the rule "these present in Post B." Negated literals are part of the update, but this trained model did not keep one. The hierarchical model and the held-out test comparison are still pending. A later vanilla run can change the number of clauses, the number of epochs, or the treatment of function words, while keeping s = 1.0 and the same evaluation split.

## References

Granmo, Ole-Christoffer. "The Tsetlin Machine – A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic." arXiv:1804.01508, 2018.

Granmo, Ole-Christoffer. "Hierarchical representation with Tsetlin machines." IKT457 Lecture 4. Implementation: https://github.com/cair/PyHierarchicalTsetlinMachineCUDA.

Lang, Ken. "NewsWeeder: Learning to Filter Netnews." *Proceedings of the 12th International Conference on Machine Learning*, 1995. The by-date 20 Newsgroups distribution used by scikit-learn is documented at http://qwone.com/~jason/20Newsgroups/.

Pedregosa, Fabian, et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research* 12 (2011): 2825–2830.

Schwartz, Martin A. "The importance of stupidity in scientific research." *Journal of Cell Science* 121 (2008): 1771. https://doi.org/10.1242/jcs.033340.

University of Agder. "Project – logical language modelling." Slide 11 of *Project - Logical Learning of Language*, IKT457.

University of Agder. *Project Report Evaluation Criteria*. IKT457 project brief.
