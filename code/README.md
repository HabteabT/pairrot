# Code

This folder is the learning code. It reads a prepared dataset and writes model results. It does not download posts, build pairs, or display a page.

## Command

From the repository root, after `dataset/prepare_data.py` has been run:

```powershell
.\.venv\Scripts\python code\train_vanilla.py --data dataset\prepared\s457_f1000_t1000_v200_e200_val0p2
```

The script loads only `features_train.npz` and `features_val.npz`. It does not read the category columns in the CSV files. It does not read the test file.

`tmu` 0.7.1 did not install here: pip stopped with "Microsoft Visual C++ 14.0 or greater is required." `code/vanilla_tsetlin.py` is the vanilla machine used for the pilot. At s = 1.0, Type I never includes a literal and always reinforces Exclude on a false literal. Type II can include a literal that is false in the current example, including a negated literal. The pilot turns `boost_true_positive_feedback` off. The library default of 1 would include true literals even when s = 1.0.

The first pilot wrote:

| File | What it is |
| --- | --- |
| `code/results/vanilla_pilot.json` | Validation accuracy 0.515, the confusion matrix, and the settings. |
| `code/results/vanilla_clauses.txt` | The included literals and two validation examples. |
| `code/results/vanilla_state.npz` | The automaton states. |
| `code/results/vanilla_status.json` | The same summary the backend can display. |

Results in `code/results/` are not committed.

The hierarchical model in `PyHierarchicalTsetlinMachineCUDA` needs an NVIDIA GPU and PyCUDA. This machine has Intel Iris Xe graphics and no `nvidia-smi`, so that model is not replaced here by another flat Tsetlin Machine.
