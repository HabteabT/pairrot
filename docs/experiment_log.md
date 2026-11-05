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
