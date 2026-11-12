# Pairrot

Four separate parts. Each one has its own folder and its own command.

| Folder | What it is | Command |
| --- | --- | --- |
| `dataset/` | Build and store the pairs. | `python dataset/prepare_data.py` |
| `code/` | Load the saved `X` and `y` for the vanilla model. | `python code/train_vanilla.py` |
| `backend/` | Serves the dashboard and its data. | Double-click `start.bat` |
| `frontend/` | The dashboard page. The backend serves it. | Open http://127.0.0.1:8000 |

## Best vanilla run so far

Double-click `run_best.bat`. It builds the pairs (full posts, 300 topical words, 30,000 training pairs), trains the vanilla machine at s = 1.0 with `--boost 1`, 800 clauses per class and T = 50, and redraws the state histogram. On a development copy of the training posts this reached validation accuracy 0.759. The experiment log entries for 2026-11-12 give every run that led there and what has to be said about the choices in the report.

Use the project virtual environment when it exists:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The report draft is in `docs/report.md`. The run log is in `docs/experiment_log.md`.

The first vanilla pilot is recorded there. Its validation accuracy was 0.515, and the only included literal was `these present in Post B`. The numbers and the clause text are in `code/results/`. The test pairs were not used.
