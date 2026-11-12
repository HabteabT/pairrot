# Pairrot experiment log

This log is part of the project record. Each meaningful run states the question first, then what changed, the settings, what happened, and what we learned. Unexpected or disappointing outcomes stay in the log. A result that is hard to interpret is still evidence about the question we asked.

The guidance for that habit comes from Martin A. Schwartz, "The importance of stupidity in scientific research," *Journal of Cell Science* 121 (2008): 1771. The useful point for this project is narrow: research often starts from not knowing, and the record should show what we learned from that, including when a run does not behave as we hoped.

## Runs

### 2026-11-05 — Can the pair builder keep the splits apart and label the pairs correctly?

- Question: if the posts are split before pairing, do the official test posts stay out, and do the saved features match the two posts?
- What changed: first implementation, then moved from the repository root into `dataset/` so it is not mixed with the model, backend, or page.
- Settings: validation 20%, vocabulary 1,000, pairs 1,000 / 200 / 200, seed 457. Recorded in `dataset/prepared/s457_f1000_t1000_v200_e200_val0p2/settings.json`.
- What happened: train 9,051, validation 2,263, test 7,532 posts. Pairs are balanced. All five data checks passed. Figshare returned HTTP 403, and the by-date archive from `qwone.com` matched scikit-learn's checksum `8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610`.
- What we learned: the most frequent vocabulary words are function words such as "the" and "to". Header, footer, and quote removal also leaves empty posts: 255 in train, 56 in validation, and 226 in test. Pair `test-00000` shows one of them. Those posts were kept because the assignment does not say to drop them.
- Command: `.\.venv\Scripts\python dataset\prepare_data.py`

### 2026-11-05 — Does a vanilla Tsetlin Machine with s=1.0 learn these pairs?

- Question: with s = 1.0, do the training pairs produce a validation accuracy and clauses we can read?
- What changed: `tmu` 0.7.1 failed to build because Microsoft Visual C++ 14.0 or newer is required. The pilot uses `code/vanilla_tsetlin.py`. The boost flag is off, so the Type I include probability is `(s - 1) / s = 0`. Four s = 1.0 checks and an illustrative one-bit problem ran before the Newsgroups fit.
- Settings: 20 clauses per class, T = 10, s = 1.0, 8 state bits, 10 epochs, seed 457, unweighted clauses. Test pairs were not loaded.
- What happened: the illustrative one-bit problem reached training accuracy 1.000. Newsgroups training accuracy finished at 0.540 and validation accuracy at 0.515. The validation confusion matrix was `[[88, 12], [85, 15]]`. Only two automata reached Include, both for `these present in Post B`. No negated literal remained. A later read of the saved state, not a new fit, found 173 of 200 validation votes tied at 0. All 27 class-1 predictions had scores `(0, 2)` and the Post B bit for "these" set. 43,922 automata were at state 0.
- What we learned: the same update learns a one-bit pattern and, on these 2,000-bit posts, mostly writes clauses and then erases them. The accuracy sits near chance because ties become class 0. The only surviving rule is a function word. The next vanilla question, still at s = 1.0, is whether more clauses or more epochs leave any other literal. The test set stays unread until that choice is made on validation.
- Command: `.\.venv\Scripts\python code\train_vanilla.py --data dataset\prepared\s457_f1000_t1000_v200_e200_val0p2`

### 2026-11-05 — Does a larger s=1.0 machine learn more than the 20-clause pilot?

- Question: if we give the same pairs more clauses and more epochs, still at s = 1.0, does validation accuracy move above the first pilot?
- What changed: 200 clauses per class, T = 50, 50 epochs. Same pairs, same vocabulary, same seed. The test pairs were still not loaded.
- What happened: training accuracy reached 0.578 at epoch 41, then the last epoch fell to 0.509. Validation accuracy finished at 0.500. The confusion matrix was `[[100, 0], [100, 0]]`, so every validation pair was predicted as a different category. The saved clauses contain 1,178 included literals and only 10 negations. Some class-1 clauses name topical words such as "crime" and "court", but their votes did not win.
- What we learned: more training at s = 1.0 wrote many more literals and still did not separate the two pair types. The final model is a constant "different category" rule. A longer run is not, by itself, a better run. The test set stays unread.

### 2026-11-12 — Is the near-chance accuracy a bug, a setting, or the task?

- Question: the two earlier runs sat at 0.515 and 0.500. Is the update code wrong, is `boost_true_positive_feedback = 0` the cause, or can a flat machine not learn "same category" from two word blocks?
- Where this ran: **not on the project machine and not on the prepared dataset.** It ran in a cloud workspace that could not download the by-date archive. The posts came from a JSON copy of the 11,314 official training posts (`selva86/datasets/newsgroups.json` on GitHub), cleaned with scikit-learn's own header, footer, and quote functions and split with the same seed. That gave 9,051 training and 2,263 validation posts, the same counts as the first run. The pairs are not the same pairs. No test posts were available, so none were read. These numbers are a development check. They should be repeated here before any of them goes into the report.
- What changed in the code: `code/vanilla_tsetlin.py` computes the clause output once per update and predicts with one matrix product. On the same seed the automaton states and predictions are identical to the old code, and training is about three times faster. `--boost 1` is a new option in `code/train_vanilla.py`. The default is still 0.
- Settings: s = 1.0 unless stated, 200 clauses per class, T = 50, seed 457. Validation used 2,000 pairs, because 200 pairs give a standard error of about 0.035 and cannot separate 0.515 from 0.500.
- What happened:

| Run | Training pairs | Train accuracy | Validation accuracy |
| --- | ---: | ---: | ---: |
| First pilot repeated: 20 clauses, T = 10, 10 epochs, 200 validation pairs | 1,000 | 0.538 | 0.515 |
| boost 0, 30 epochs | 1,000 | 0.546 | 0.492 |
| boost 1, 30 epochs | 1,000 | 0.990 | 0.489 |
| boost 0, 8 epochs | 10,000 | 0.534 | 0.521 |
| boost 1, 8 epochs | 10,000 | 0.730 | 0.512 |
| boost 1, 8 epochs, words in more than 10% of posts removed | 10,000 | 0.745 | 0.517 |
| the same, at most 32 literals per clause | 10,000 | 0.711 | 0.505 |
| diagnostic, s = 3, boost 1, 6 epochs (not an assignment setting) | 10,000 | 0.570 | 0.506 |
| diagnostic, s = 10, boost 1, 6 epochs (not an assignment setting) | 10,000 | 0.571 | 0.510 |

  With boost 1 the machine kept about 27,000 literals after 1,000 pairs, and all but a handful were negated. With boost 0 it kept 13. Two reference models that are not Tsetlin Machines were fit on 10,000 to 20,000 pairs: logistic regression on the raw Post A and Post B blocks reached 0.513 on validation, and logistic regression on "the same word is in both posts" reached 0.55 to 0.60 for every vocabulary tried (top 1,000 or 5,000 words, with or without the most frequent words).
- What we learned: the update code is not the cause. It reproduces the first pilot exactly, and with boost 1 it fits the training pairs almost perfectly. The boost flag decides whether the machine can memorise, not whether it generalises: boost 1 at s = 1.0 learns each training pair by listing absent words, which is reasoning by elimination, and that list does not transfer to new pairs. Ten times more pairs, a vocabulary without function words, a clause-size limit, and ordinary s values all stayed within about two standard errors of 0.500. The label depends only on how Post A and Post B relate, so no single word in one block carries any signal, and a flat conjunction has to find the right word pair among a million. Even the reference that is handed the "same word in both posts" feature stops near 0.60. A vanilla accuracy near 0.5 is therefore the measured answer for this encoding, not a defect to tune away. The comparison with the hierarchical model is where a difference should appear.
- Open point for the report: tmu and pyTsetlinMachine default to boost 1. The report explains why the pilot uses 0. Both readings of s = 1.0 give the same validation accuracy here, and they give very different clauses, which is worth showing.
- Commands: `python code/train_vanilla.py --boost 1`, and `python dataset/prepare_data.py --val-pairs 2000 --train-pairs 10000` for the larger pair sets.

### 2026-11-12 — Does keeping the full posts help, and why does s = 1.0 stay near 0.5?

- Question: the course brief asks for 20 Newsgroups, two sets of words, and a vanilla machine with small s (= 1.0). It does not ask for headers, footers, and quotes to be removed. If they are kept, does the vanilla machine do better?
- Where this ran: the same development copy of the official training posts as the entry above, not the project machine. No test posts.
- What changed: `dataset/prepare_data.py --full-text` keeps the posts as distributed. The default still removes headers, footers, and quotes. For this check the vocabulary was the 1,000 most frequent words that occur in at most 10% of the training posts.
- Settings: 10,000 training pairs, 2,000 validation pairs, 200 clauses per class, T = 50, s = 1.0, 6 epochs, seed 457.
- What happened: no post was empty. Boost 0 reached validation accuracy 0.516. Boost 1 reached 0.537 with training accuracy 0.726. The logistic-regression reference on "the same word is in both posts" rose from about 0.58 to 0.65.
- What we learned: full posts carry more signal, and boost 1 picks up a little of it: 0.537 is about three standard errors above 0.500. With boost 0 there is a structural reason for staying at chance. At s = 1.0 a clause that does not fire on a target-class pair has every literal pushed toward Exclude, and Type I never includes anything, so a clause specific to one category, such as "hockey in A and hockey in B", fires on about one same-category pair in twenty and is erased. This entry first concluded that the same limit holds with boost 1. The next entry shows that conclusion was too strong: with a topical vocabulary, boost 1 does generalise.
- Command: `python dataset/prepare_data.py --full-text --train-pairs 10000 --val-pairs 2000`, then `python code/train_vanilla.py --boost 1 --data dataset/prepared/<the new folder>`.

### 2026-11-12 — Can a topical vocabulary take the s = 1.0 machine past 0.55?

- Question: boost 1 learns by listing absent words. With a vocabulary of frequent words that list did not transfer. If the vocabulary holds the words that separate the categories, does elimination start to generalise?
- Where this ran: the development copy of the official training posts, not the project machine. No test posts. One number was repeated with the project's own `code/train_vanilla.py`.
- What changed: `dataset/prepare_data.py --vocabulary topical` scores each word by the chi-square statistic between its presence and the category, on training posts only, without English stop words and without words in fewer than five training posts. The categories now choose the vocabulary as well as building Y. They are still not features, and validation posts are not used for the choice. The default vocabulary is unchanged.
- Settings: 1,000 words, 10,000 training pairs, s = 1.0, boost 1, 200 clauses per class, T = 50, seed 457. Validation is 2,000 pairs. "Check" is a second set of 2,000 pairs from the validation posts, drawn with another seed and never used to pick anything.
- What happened:

| Run | Train | Validation | Check |
| --- | ---: | ---: | ---: |
| Cleaned posts, 8 epochs | 0.775 | 0.546 | 0.561 |
| Full posts, 8 epochs | 0.808 | 0.593 | 0.599 |
| Full posts, 8 epochs, project script (fixed pair order) | 0.812 | 0.589 | not scored |
| Full posts, 8 epochs, 800 clauses, T = 200 | 0.807 | 0.607 | 0.614 |
| Full posts, 24 epochs | 0.847 | 0.617 | 0.623 |

  Every included literal was negated: about 144,000 of them in 400 clauses after 24 epochs, roughly 360 absent words per clause. The logistic-regression reference on "the same word is in both posts" reached 0.678 with full posts and 0.597 with cleaned posts. With cleaned posts about 14% of pairs have a side with none of the 1,000 words. With full posts it is under 1%.
- What we learned: the vanilla machine at s = 1.0 can pass 0.55, and it does so by elimination. A clause that lists the topical words of many categories as absent from both posts is left with pairs that share the remaining topics. That only works when the vocabulary is topical and when Type I can include true literals, which is what boost 1 does. The earlier runs failed for two separate reasons: boost 0 cannot build such a clause, and a frequent-word vocabulary gives it nothing to eliminate. Full posts matter because a cleaned post often contains none of the words. Accuracy was still rising at 24 epochs.
- What to say in the report: the vocabulary was chosen with the category labels of the training posts, the posts were not cleaned, and boost was 1. All three are choices, not requirements of the brief.
- Commands: `python dataset/prepare_data.py --full-text --vocabulary topical --train-pairs 10000 --val-pairs 2000`, then `python code/train_vanilla.py --boost 1 --epochs 24 --data dataset/prepared/s457_f1000_t10000_v2000_e200_val0p2_fulltext_topical`.

### 2026-11-12 — How far does elimination go with more pairs and a tuned vocabulary?

- Question: the topical vocabulary reached about 0.60 with 10,000 pairs and 1,000 words. Which of pair count, vocabulary size, clause count, and T moves it further?
- Where this ran: the development copy of the official training posts, not the project machine. No test posts. The last row is the project's own `code/train_vanilla.py`.
- What changed: only settings. Full posts, topical vocabulary, s = 1.0, boost 1, seed 457 throughout. `run_best.bat` now runs the last row from start to finish.
- What happened. Validation is 2,000 pairs. Check is a second 2,000 pairs from the validation posts that was never used to choose a setting. Each row reports the last epoch, not the best one.

| Words | Training pairs | Clauses per class | T | Epochs | Train | Validation | Check |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 10,000 | 200 | 50 | 24 | 0.847 | 0.617 | 0.623 |
| 2,000 | 10,000 | 200 | 50 | 16 (stopped) | 0.873 | 0.554 | not scored |
| 1,000 | 30,000 | 200 | 50 | 12 | 0.745 | 0.651 | not scored |
| 1,000 | 100,000 | 200 | 50 | 4 | not recorded | 0.661 | not scored |
| 500 | 30,000 | 200 | 50 | 12 | 0.740 | 0.691 | not scored |
| 500 | 30,000 | 200 | 100 | 10 | not recorded | 0.666 | 0.658 |
| 500 | 30,000 | 200 | 25 | 10 | not recorded | 0.693 | 0.685 |
| 300 | 30,000 | 200 | 50 | 12 | not recorded | 0.698 | 0.704 |
| 300 | 30,000 | 200 | 25 | 12 | not recorded | 0.720 | 0.710 |
| 300 | 30,000 | 400 | 50 | 12 | not recorded | 0.731 | 0.722 |
| 200 | 30,000 | 400 | 25 | 12 | 0.787 | 0.733 | 0.742 |
| 300 | 30,000 | 800 | 50 | 8 | 0.802 | 0.744 | 0.742 |
| 300 | 30,000 | 800 | 50 | 8, project script | 0.799 | 0.759 | 0.766 |

  In the last row the validation confusion matrix was `[[871, 129], [353, 647]]`: 87% of different-category pairs and 65% of same-category pairs were right. All 339,815 included literals were negated, and all 1,600 clauses held literals.
- What we learned: three things moved the result. More pairs helped up to 30,000 and not beyond at this clause count. A smaller, more topical vocabulary helped: 300 words beat 1,000, and 2,000 was worse than 1,000. A low T relative to the clause count helped, and more clauses helped once T was low. The gap between training and validation accuracy shrank from 0.23 to 0.04 along the way, so the first topical runs were memorising pairs and the last one is not. The model is better at saying "different" than "same". With 200 words about 24% of validation pairs have a post with none of the words, so a smaller vocabulary than 300 trades coverage for clarity.
- Caution: the settings were chosen on the validation pairs. The check pairs agree with validation, which says the choice did not overfit those 2,000 pairs, but check and validation share the same posts. The official test pairs are the only clean number, and they have not been read.
- Command: double-click `run_best.bat`, or run its three commands by hand.
