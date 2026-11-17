# Hierarchical Tsetlin Machine

This is the second part of the assignment. It uses the same pairs as the vanilla model.

- Dataset: 20 Newsgroups, all 20 categories.
- Each example is two posts. The feature row is the words of Post A, then the words of Post B.
- Y = 1 when the two posts are the same category. Y = 0 when they are different.
- Each clause is a tree. One group reads Post A and one group reads Post B. A group adds its leaf conjunctions, and the clause multiplies the two groups. s = 1.0 and boost is off, the same elimination rule as the vanilla machine.
- The official test pairs are not read.

`PyHierarchicalTsetlinMachineCUDA` needs an NVIDIA GPU. This computer does not have one, so `hierarchical_tsetlin.py` trains that same tree in Python. It is not the CUDA program, and it is not the flat vanilla machine.

## Run

```powershell
.\.venv\Scripts\python.exe hierarchical\train_hierarchical.py
```

Or double-click `hierarchical\train.bat`. The result is written to `hierarchical/results/hierarchical.json`. Compare its validation accuracy with the vanilla model in `code/results/vanilla_pilot.json`.
