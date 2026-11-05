"""HTTP API for Pairrot.

The backend reads prepared dataset files and model status files.
It does not download posts, sample pairs, or train a Tsetlin Machine.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


REPO = Path(__file__).resolve().parents[1]
PREPARED = REPO / "dataset" / "prepared"
MODEL_STATUS = REPO / "code" / "results" / "vanilla_status.json"
FRONTEND = REPO / "frontend"
HOST = "127.0.0.1"
PORT = 8000
PAGES = {
    "/": FRONTEND / "index.html",
    "/index.html": FRONTEND / "index.html",
    "/app.js": FRONTEND / "app.js",
    "/styles.css": FRONTEND / "styles.css",
}
PAGE_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


def latest_prepared_dir() -> Path | None:
    runs = [path for path in PREPARED.iterdir() if (path / "settings.json").exists()] if PREPARED.exists() else []
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def all_pairs(data_dir: Path, split: str) -> list[dict[str, str]]:
    if split not in {"train", "val", "test"}:
        raise FileNotFoundError(split)
    path = data_dir / f"pairs_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(split)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    positive = sum(1 for row in rows if row["y"] == "1")
    return {"y1": positive, "y0": len(rows) - positive}


def dataset_payload(limit: int, split: str) -> dict:
    data_dir = latest_prepared_dir()
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
            name: label_counts(all_pairs(data_dir, name)) for name in ("train", "val", "test")
        },
        "pairs": all_pairs(data_dir, split)[:limit],
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
    if "accuracy" in status["status"]:
        return {"state": "ready", "detail": f"Validation accuracy {status['status']['accuracy']}."}
    return {"state": "waiting", "detail": f"Latest status file says {stage}."}


def hierarchical_stage() -> dict[str, str]:
    if shutil.which("nvidia-smi"):
        return {
            "state": "waiting",
            "detail": "An NVIDIA driver command is available. The hierarchical model has not been trained.",
        }
    return {
        "state": "blocked",
        "detail": "nvidia-smi was not found. The Lecture 4 model needs an NVIDIA GPU through CUDA.",
    }


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
    }


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

    def _send_page(self, path: str) -> None:
        page = PAGES.get(path)
        if page is None or not page.exists():
            self._send({"error": "Not found"}, status=404)
            return
        body = page.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", PAGE_TYPES[page.suffix])
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
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open the dashboard at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
