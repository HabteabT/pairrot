# Pairrot

Four separate parts. Each one has its own folder and its own command.

| Folder | What it is | Command |
| --- | --- | --- |
| `dataset/` | Build and store the pairs. | `python dataset/prepare_data.py` |
| `code/` | Load the saved `X` and `y` for the vanilla model. | `python code/train_vanilla.py` |
| `hierarchical/` | Train the Lecture 4 hierarchical model on the same pairs. | `python hierarchical/train_hierarchical.py` |
| `backend/` | Serves the dashboard and its data. | Double-click `start.bat` |
| `frontend/` | The dashboard page. The backend serves it. | Open http://127.0.0.1:8000 |

## Required vanilla run

Double-click `train.bat`. It trains on the cleaned pairs in `dataset/prepared/s457_f1000_t10000_v2000_e2000_val0p2`: headers, footers, and quotes removed, 1,000 frequent words, s = 1.0, boost off. The test pairs are not scored.

`run_best.bat` is a separate experiment. It keeps the headers, chooses words with the category labels, and turns boost on. Those three choices are not part of the required run.

Use the project virtual environment when it exists:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The dataset catalog is in `docs/datasets.md`. The report draft is in `docs/report.md`. The run log is in `docs/experiment_log.md`.

On the cleaned 10,000-pair dataset, s = 1.0, the dashboard shows the boost-on run: vanilla validation 0.521 and hierarchical validation 0.529. The same pairs with boost off scored 0.517 and 0.501. Those files are in `code/results/archive/20261117-224250_val0.517` and `hierarchical/results/archive/boost0_val0.501`. The test pairs were not used. The comparison is written in `docs/datasets.md`.
