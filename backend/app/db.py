"""SQLite persistence for benchmark runs (stdlib ``sqlite3`` only).

One row per transcription so RTF / latency / WER can be analysed later. The
connection is shared across threads: SQLite is opened with
``check_same_thread=False`` and every statement is serialised behind a
:class:`threading.Lock`, which is plenty for a single-container POC.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 REAL    NOT NULL,
    model              TEXT    NOT NULL,
    mode               TEXT    NOT NULL,
    audio_ms           REAL,
    processing_ms      REAL,
    rtf                REAL,
    words              INTEGER,
    chars              INTEGER,
    latency_partial_ms REAL,
    latency_final_ms   REAL,
    text               TEXT,
    text_hash          TEXT,
    expected_text      TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_ts    ON runs (ts DESC);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs (model, ts DESC);
"""

_COLUMNS = (
    "ts",
    "model",
    "mode",
    "audio_ms",
    "processing_ms",
    "rtf",
    "words",
    "chars",
    "latency_partial_ms",
    "latency_final_ms",
    "text",
    "text_hash",
    "expected_text",
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that post-date the original CREATE TABLE.

    ``CREATE TABLE IF NOT EXISTS`` is a no-op on a database created before a
    column was added, so new columns need an explicit ALTER.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    for column, ddl in (("expected_text", "TEXT"),):
        if column not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {column} {ddl}")


class Database:
    """Thread-safe wrapper around a single SQLite connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                if self.path.parent and str(self.path) != ":memory:":
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(self.path), check_same_thread=False)
                conn.row_factory = sqlite3.Row
                # WAL keeps readers (GET /runs) from blocking the writer.
                if str(self.path) != ":memory:":
                    conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.executescript(SCHEMA)
                _migrate(conn)
                conn.commit()
                self._conn = conn
            return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # -- writes -------------------------------------------------------------
    def insert_run(
        self,
        *,
        model: str,
        mode: str,
        audio_ms: float | None = None,
        processing_ms: float | None = None,
        rtf: float | None = None,
        words: int | None = None,
        chars: int | None = None,
        latency_partial_ms: float | None = None,
        latency_final_ms: float | None = None,
        text: str | None = None,
        text_hash: str | None = None,
        expected_text: str | None = None,
        ts: float | None = None,
    ) -> int:
        """Insert one run and return its rowid."""
        values: tuple[Any, ...] = (
            time.time() if ts is None else ts,
            model,
            mode,
            audio_ms,
            processing_ms,
            rtf,
            words,
            chars,
            latency_partial_ms,
            latency_final_ms,
            text,
            text_hash,
            expected_text,
        )
        placeholders = ", ".join("?" * len(_COLUMNS))
        sql = f"INSERT INTO runs ({', '.join(_COLUMNS)}) VALUES ({placeholders})"
        conn = self.connect()
        with self._lock:
            cur = conn.execute(sql, values)
            conn.commit()
            return int(cur.lastrowid or 0)

    # -- reads --------------------------------------------------------------
    def recent_runs(
        self,
        limit: int = 50,
        model: str | None = None,
        mode: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT id, " + ", ".join(_COLUMNS) + " FROM runs"
        clauses: list[str] = []
        params: list[Any] = []
        if model:
            clauses.append("model = ?")
            params.append(model)
        if mode:
            clauses.append("mode = ?")
            params.append(mode)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(int(limit))

        conn = self.connect()
        with self._lock:
            rows: Iterable[sqlite3.Row] = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def count_runs(self) -> int:
        conn = self.connect()
        with self._lock:
            return int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])


# Process-wide instance used by the routes.
db = Database(settings.db_path)
