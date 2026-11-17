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

The dashboard shows the flexible run: full posts, 300 topical words, s = 1.0, boost on. Vanilla validation is 0.711. The second Tsetlin level, trained on those clause outputs and on shared words, is 0.726. The required cleaned pairs stay in the comparison: boost off is 0.517 and 0.501, boost on is 0.523 and 0.529. Those files are archived. The test pairs were not used. The catalog is `docs/datasets.md`.
