"""SQLite storage for sessions and messages.

The schema is intentionally small. Two tables:

  sessions(id, title, created_at, updated_at, current_model, system_prompt)
  messages(id, session_id, role, content, model, created_at)

Every user prompt and assistant response is persisted immediately on arrival,
so a crash, kill, or power loss never costs more than the in-flight token.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


DEFAULT_DB_PATH = Path(os.environ.get("LLMCHAT_DB",
                                      Path.home() / ".llmchat" / "sessions.db"))


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    current_model TEXT NOT NULL,
    system_prompt TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content    TEXT NOT NULL,
    model      TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
"""


@dataclass
class Session:
    id: str
    title: str
    created_at: float
    updated_at: float
    current_model: str
    system_prompt: Optional[str]


@dataclass
class Message:
    id: int
    session_id: str
    role: str
    content: str
    model: Optional[str]
    created_at: float


class Store:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        # WAL gives us concurrent readers + better crash safety.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    # ----- sessions ---------------------------------------------------------

    def create_session(self, model: str, title: str = "New chat",
                       system_prompt: Optional[str] = None) -> Session:
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at, "
                "current_model, system_prompt) VALUES (?, ?, ?, ?, ?, ?)",
                (sid, title, now, now, model, system_prompt),
            )
        return Session(sid, title, now, now, model, system_prompt)

    def get_session(self, session_id: str) -> Optional[Session]:
        # Allow prefix matches so users can type just the first few chars.
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ? OR id LIKE ? "
                "ORDER BY updated_at DESC LIMIT 1",
                (session_id, session_id + "%"),
            ).fetchone()
        return _row_to_session(row) if row else None

    def list_sessions(self, limit: int = 50) -> list[Session]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_session(r) for r in rows]

    def update_session_model(self, session_id: str, model: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET current_model = ?, updated_at = ? "
                "WHERE id = ?",
                (model, time.time(), session_id),
            )

    def update_session_title(self, session_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, time.time(), session_id),
            )

    def update_system_prompt(self, session_id: str,
                             system_prompt: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET system_prompt = ?, updated_at = ? "
                "WHERE id = ?",
                (system_prompt, time.time(), session_id),
            )

    def delete_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # ----- messages ---------------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str,
                    model: Optional[str] = None) -> Message:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages(session_id, role, content, model, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, model, now),
            )
            msg_id = cur.lastrowid
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
        return Message(msg_id, session_id, role, content, model, now)

    def get_messages(self, session_id: str) -> list[Message]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [_row_to_message(r) for r in rows]


def _row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        current_model=row["current_model"],
        system_prompt=row["system_prompt"],
    )


def _row_to_message(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        model=row["model"],
        created_at=row["created_at"],
    )
