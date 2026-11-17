"""HTTP API for Pairrot.

The backend reads prepared dataset files and model status files.
It does not download posts, sample pairs, or train a Tsetlin Machine.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


REPO = Path(__file__).resolve().parents[1]
PREPARED = REPO / "dataset" / "prepared"
MODEL_STATUS = REPO / "code" / "results" / "vanilla_status.json"
PROGRESS = REPO / "code" / "results" / "vanilla_progress.json"
FRONTEND = REPO / "frontend"
CODE = REPO / "code"
_MEASURED: dict[str, str] = {}
HOST = "127.0.0.1"
PORT = 8000
FIGURE = CODE / "results" / "knows_when.png"
LINE = CODE / "results" / "accuracy_line.png"
PAGES = {
    "/": FRONTEND / "index.html",
    "/index.html": FRONTEND / "index.html",
    "/app.js": FRONTEND / "app.js",
    "/styles.css": FRONTEND / "styles.css",
    "/knows_when.png": FIGURE,
    "/accuracy_line.png": LINE,
}
PAGE_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
}


def latest_prepared_dir() -> Path | None:
    runs = [path for path in PREPARED.iterdir() if (path / "settings.json").exists()] if PREPARED.exists() else []
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def all_pairs(data_dir: Path, split: str, limit: int | None = None) -> list[dict[str, str]]:
    if split not in {"train", "val", "test"}:
        raise FileNotFoundError(split)
    path = data_dir / f"pairs_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(split)
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    positive = sum(1 for row in rows if row["y"] == "1")
    return {"y1": positive, "y0": len(rows) - positive}


def shown_progress() -> dict | None:
    """Live epoch file, used only while the saved run has not finished."""

    if not PROGRESS.exists():
        return None
    try:
        progress = read_json(PROGRESS)
    except json.JSONDecodeError:
        return None
    if not progress.get("data_dir"):
        return None
    if MODEL_STATUS.exists():
        try:
            saved = read_json(MODEL_STATUS)
        except json.JSONDecodeError:
            saved = {}
        if saved.get("run_complete"):
            return None
        if int(progress.get("epoch") or 0) <= int(saved.get("epochs_finished") or 0):
            return None
    return progress


def model_data_dir() -> Path | None:
    """Use the dataset behind the score on the page, so the table matches it."""

    progress = shown_progress()
    if progress is not None:
        recorded = Path(progress["data_dir"])
        if (recorded / "settings.json").exists():
            return recorded
    if MODEL_STATUS.exists():
        recorded = read_json(MODEL_STATUS).get("data_dir")
        if recorded and (Path(recorded) / "settings.json").exists():
            return Path(recorded)
    return latest_prepared_dir()


def dataset_payload(limit: int, split: str) -> dict:
    data_dir = model_data_dir()
    if data_dir is None:
        return {"ready": False, "message": "No prepared dataset. Run dataset/prepare_data.py."}
    settings = read_json(data_dir / "settings.json")
    vocabulary = read_json(data_dir / "vocabulary.json")
    words = vocabulary["words"][:12]
    frequencies = vocabulary["document_frequency"][:12]
    return {
        "ready": True,
        "prepared_dir": data_dir.name,
        "documents_per_split": settings["measured"]["documents_per_split"],
        "pairs_per_split": settings["measured"]["pairs_per_split"],
        "empty_documents_after_tokenization": settings["measured"]["empty_documents_after_tokenization"],
        "vocabulary_size": settings["measured"]["vocabulary_size"],
        "feature_width": settings["measured"]["feature_width"],
        "top_words": [
            {"word": word, "document_frequency": frequency}
            for word, frequency in zip(words, frequencies)
        ],
        "label_counts": {
            name: {
                "y1": settings["measured"]["pairs_per_split"][name] // 2,
                "y0": settings["measured"]["pairs_per_split"][name] // 2,
            }
            for name in ("train", "val", "test")
        },
        "pairs": all_pairs(data_dir, split, limit),
    }


def vanilla_stage(status: dict) -> dict[str, str]:
    if not status["ready"]:
        return {"state": "waiting", "detail": status["message"]}
    stage = status["status"].get("stage", "unknown")
    if stage == "vanilla-not-trained":
        shape = " x ".join(str(part) for part in status["status"].get("train_shape", []))
        return {
            "state": "waiting",
            "detail": f"Training features are loaded ({shape}). The Tsetlin Machine has not been trained.",
        }
    progress = shown_progress()
    if progress is not None:
        score = float(progress["validation_accuracy"])
        return {"state": "ready", "detail": f"Validation accuracy {score:.3f} ({score * 100:.1f}%)."}
    if "accuracy" in status["status"]:
        score = float(status["status"].get("validation_accuracy", status["status"]["accuracy"]))
        return {"state": "ready", "detail": f"Validation accuracy {score:.3f} ({score * 100:.1f}%)."}
    return {"state": "waiting", "detail": f"Latest status file says {stage}."}


def hierarchical_stage() -> dict[str, str]:
    folder = REPO / "hierarchical" / "results"
    saved: dict = {}
    final_path = folder / "hierarchical.json"
    if final_path.exists():
        saved = json.loads(final_path.read_text(encoding="utf-8"))
    progress: dict = {}
    progress_path = folder / "hierarchical_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    training = bool(progress) and not progress.get("run_complete") and not saved.get("run_complete")
    if training and progress.get("epoch", 0) >= saved.get("best_epoch", 0):
        score = float(progress.get("best_validation", progress.get("validation", 0)))
        return {
            "state": "ready",
            "detail": (
                f"Training. Best validation so far {score:.3f} ({score * 100:.1f}%). "
                f"Epoch {progress.get('epoch')}."
            ),
        }
    if saved.get("validation_accuracy") is not None:
        score = float(saved["validation_accuracy"])
        return {"state": "ready", "detail": f"Validation accuracy {score:.3f} ({score * 100:.1f}%)."}
    return {"state": "waiting", "detail": "The hierarchical model has not been trained yet."}


def run_output() -> str:
    """Score the saved machine once, then reuse that printout until the files change."""

    progress = shown_progress()
    if progress is not None:
        hyper = progress.get("hyperparameters") or {}
        lines = [
            f"Training is still running. Epoch {progress.get('epoch')} is the latest finished epoch.",
            Path(progress["data_dir"]).name,
            (
                f"clauses {hyper.get('clauses_per_class')}, T {hyper.get('T')}, "
                f"s {hyper.get('s')}, boost {hyper.get('boost_true_positive_feedback')}"
            ),
            "",
        ]
        for row in progress.get("epoch_history") or []:
            lines.append(
                f"epoch {row['epoch']}: train {row['train']:.3f}, validation {row['validation']:.3f}"
            )
        score = float(progress["validation_accuracy"])
        lines.extend(["", f"Latest validation accuracy {score:.3f} ({score * 100:.1f}%)."])
        return "\n".join(lines)

    state_path = CODE / "results" / "vanilla_state.npz"
    if not state_path.exists():
        return "No saved model yet. Double-click train.bat, then refresh this page."
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    try:
        from measure import measure, trained_data_dir

        # Score on the dataset the model was trained on. The newest prepared
        # folder is only a fallback, because its vocabulary can be different.
        data_dir = trained_data_dir(CODE / "results") or latest_prepared_dir()
        if data_dir is None:
            return "A saved model exists, but no prepared dataset was found. Run dataset/prepare_data.py."
        stamp = f"{state_path.stat().st_mtime}:{data_dir}:{data_dir.stat().st_mtime}"
        if _MEASURED.get("stamp") != stamp:
            _MEASURED["text"] = measure(data_dir, CODE / "results")
            _MEASURED["stamp"] = stamp
        return _MEASURED["text"]
    except Exception as error:  # The page should say what went wrong, not go blank.
        return f"Could not score the saved model: {type(error).__name__}: {error}"


def dashboard_payload(split: str, limit: int) -> dict:
    dataset = dataset_payload(limit, split)
    vanilla = model_payload()
    dataset_state = {
        "state": "ready" if dataset["ready"] else "waiting",
        "detail": (
            f"{dataset['prepared_dir']}: {dataset['pairs_per_split']['train']} training pairs."
            if dataset["ready"]
            else dataset["message"]
        ),
    }
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "stages": [
            {"id": "dataset", "title": "Dataset", **dataset_state},
            {"id": "vanilla", "title": "Vanilla Tsetlin Machine", **vanilla_stage(vanilla)},
            {"id": "hierarchical", "title": "Hierarchical Tsetlin Machine", **hierarchical_stage()},
        ],
        "dataset": dataset,
        "vanilla": vanilla,
        "run_output": run_output(),
    }


def ensure_accuracy_line() -> None:
    """Redraw the chart when the run on the page is newer than the picture."""

    source = PROGRESS if shown_progress() is not None else CODE / "results" / "vanilla_pilot.json"
    if not source.exists():
        return
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    from accuracy_line import load_history

    if not load_history():
        if LINE.exists():
            LINE.unlink()
        return
    sources = [source]
    hierarchical = REPO / "hierarchical" / "results" / "hierarchical.json"
    if hierarchical.exists():
        sources.append(hierarchical)
    newest = max(item.stat().st_mtime for item in sources)
    if LINE.exists() and LINE.stat().st_mtime >= newest:
        return
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    from accuracy_line import main as draw_accuracy_line

    draw_accuracy_line()


def model_payload() -> dict:
    if not MODEL_STATUS.exists():
        return {
            "ready": False,
            "message": "No model status. The training command is python code/train_vanilla.py.",
        }
    return {"ready": True, "status": read_json(MODEL_STATUS)}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self._send_page(parsed.path)
            return
        query = parse_qs(parsed.query)
        split = query.get("split", ["train"])[0]
        if split not in {"train", "val", "test"}:
            self._send({"error": "split must be train, val, or test"}, status=400)
            return
        try:
            limit = int(query.get("limit", ["6"])[0])
        except ValueError:
            self._send({"error": "limit must be an integer"}, status=400)
            return
        limit = min(max(limit, 1), 20)
        try:
            if parsed.path == "/api/dashboard":
                self._send(dashboard_payload(split, limit))
            elif parsed.path == "/api/dataset":
                self._send(dataset_payload(limit, split))
            elif parsed.path == "/api/model":
                self._send(model_payload())
            elif parsed.path == "/api/health":
                self._send({"ok": True, "service": "pairrot-backend"})
            else:
                self._send({"error": "Not found"}, status=404)
        except FileNotFoundError:
            self._send({"error": f"No pairs found for split {split}."}, status=404)
        except Exception as error:
            # A half-written or damaged results file should not drop the connection.
            self._send({"error": f"{type(error).__name__}: {error}"}, status=500)

    def _send_page(self, path: str) -> None:
        if path == "/accuracy_line.png":
            ensure_accuracy_line()
        page = PAGES.get(path)
        if page is None or not page.exists():
            self._send({"error": "Not found"}, status=404)
            return
        body = page.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", PAGE_TYPES[page.suffix])
        # The figures are redrawn after every training run, so never reuse an old copy.
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[backend] {self.address_string()} {format % args}")


def main() -> None:
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as error:
        raise SystemExit(
            f"Could not start on http://{HOST}:{PORT} ({error}). "
            "The dashboard is probably already running in another window. Close it and try again."
        )
    print(f"Open the dashboard at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
