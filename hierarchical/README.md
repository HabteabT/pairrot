# Hierarchical Tsetlin Machine

This is the second part of the assignment. It uses the same pairs as the vanilla model.

- Dataset: 20 Newsgroups, all 20 categories.
- Each example is two posts. The feature row is the words of Post A, then the words of Post B.
- Y = 1 when the two posts are the same category. Y = 0 when they are different.
- The first level is the vanilla Tsetlin Machine. The second level is another Tsetlin Machine trained on those clause outputs and on words that appear in both posts. s = 1.0. Boost is on for the second level.
- The official test pairs are not read.

The saved result is that second Tsetlin machine. `hierarchical_tsetlin.py` still holds a Python tree for the Lecture 4 shape. `PyHierarchicalTsetlinMachineCUDA` was not run. This computer has no NVIDIA GPU.

## Run

```powershell
.\.venv\Scripts\python.exe hierarchical\train_hierarchical.py
```

Or double-click `hierarchical\train.bat`. The result is written to `hierarchical/results/hierarchical.json`. Compare its validation accuracy with the vanilla model in `code/results/vanilla_pilot.json`.
