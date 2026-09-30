"""Optional local conversation saving as JSON files.

Files land in conversations/ (configurable via N_ATLAS_CONVERSATIONS_DIR) as:

    conversations/2026-09-30_15-30-00.json

Each file stores timestamp, model and messages. The API key, tunnel URL and
any other server credentials are never written to these files.
"""

import json
import time
from pathlib import Path


def ensure_dir(directory) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_filename() -> str:
    return "conversation_%s.json" % time.strftime("%Y-%m-%d_%H-%M-%S")


def save_conversation_to_disk(cfg, messages, directory=None):
    """Write messages to conversations/<timestamp>.json. Returns the path."""
    directory_path = ensure_dir(directory or cfg.conversations_dir)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": cfg.model,
        "messages": [dict(message) for message in messages],
    }
    path = directory_path / default_filename()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def load_conversation_from_disk(path):
    """Load a conversation file. Returns (messages, model)."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    messages = payload.get("messages", [])
    model = payload.get("model", "")
    return (
        [{"role": m.get("role", "user"), "content": m.get("content", "")}
         for m in messages],
        model,
    )


def list_conversations(directory="conversations"):
    """Return sorted JSON conversation files (newest first)."""
    path = Path(directory)
    if not path.exists():
        return []
    files = sorted(
        path.glob("conversation_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files