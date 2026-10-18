"""Chat session history storage for DocuMind AI."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

CHATS_DIR = Path(__file__).parent / ".chats"


def _ensure_dir() -> Path:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    return CHATS_DIR


def list_sessions() -> List[Dict[str, Any]]:
    """Return all saved sessions sorted by last updated descending."""
    _ensure_dir()
    sessions: List[Dict[str, Any]] = []
    for file_path in CHATS_DIR.glob("*.json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            sessions.append(
                {
                    "id": data.get("id", file_path.stem),
                    "title": data.get("title", "Untitled Chat"),
                    "doc_name": data.get("doc_name"),
                    "updated_at": data.get("updated_at", ""),
                    "message_count": len(data.get("messages", [])),
                }
            )
        except Exception:
            continue

    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return sessions


def load_session(session_id: str) -> Dict[str, Any] | None:
    """Load session data by id."""
    file_path = CHATS_DIR / f"{session_id}.json"
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_session(
    session_id: str,
    messages: List[Dict[str, Any]],
    doc_name: str | None = None,
    title: str | None = None,
) -> str:
    """Save or update a conversation session. Returns session_id."""
    _ensure_dir()
    file_path = CHATS_DIR / f"{session_id}.json"

    existing: Dict[str, Any] = {}
    if file_path.exists():
        try:
            existing = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    if not title:
        # Determine title from existing or first user message
        if existing.get("title") and existing["title"] != "New Conversation":
            title = existing["title"]
        else:
            first_user = next((m["content"] for m in messages if m.get("role") == "user"), None)
            if first_user:
                cleaned = " ".join(first_user.split())
                title = cleaned[:35] + ("…" if len(cleaned) > 35 else "")
            else:
                title = f"Document: {doc_name}" if doc_name else "New Conversation"

    data = {
        "id": session_id,
        "title": title,
        "doc_name": doc_name if doc_name is not None else existing.get("doc_name"),
        "created_at": existing.get("created_at", datetime.now().isoformat()),
        "updated_at": datetime.now().isoformat(),
        "messages": messages,
    }

    file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return session_id


def delete_session(session_id: str) -> bool:
    """Delete a session file by id."""
    file_path = CHATS_DIR / f"{session_id}.json"
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def create_new_session_id() -> str:
    """Generate a unique session identifier."""
    return str(uuid.uuid4())

