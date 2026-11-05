# Backend

This folder serves the dashboard and the JSON behind it. It does not prepare the dataset and it does not train the model.

## Command

Double-click `start.bat` in the Pairrot folder, or from the repository root run:

```powershell
.\.venv\Scripts\python backend\app.py
```

Then open http://127.0.0.1:8000.

| Route | Reads |
| --- | --- |
| `GET /api/dashboard` | Dataset status, vanilla status, and whether an NVIDIA driver is visible. |
| `GET /api/health` | Nothing. |
| `GET /api/dataset?split=train&limit=4` | `dataset/prepared/...` |
| `GET /api/model` | `code/results/vanilla_status.json`, if that file exists. |

The same process also serves `frontend/index.html`, `frontend/app.js`, and `frontend/styles.css`.
