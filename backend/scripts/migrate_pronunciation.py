#!/usr/bin/env python3
"""One-off: rewrite `expected_text` ground truth to the spoken form of a name.

The speaker's name is written "Dhira" but pronounced "Dira" (silent h). Ground
truth captured with the orthographic spelling charged one substitution on every
utterance containing the name, so historical WER is overstated — run #17 scored
12.5% purely from that word on an otherwise perfect transcript.

`PRONUNCIATION_EQUIV` in `app.metrics` fixes this going forward for *scoring*,
but the stored ground truth is what a human reads in the Reports table, so it is
corrected here too.

Idempotent: running it twice changes nothing the second time. Takes a timestamped
backup next to the database before writing, and only ever touches `expected_text`.

Usage:
    python scripts/migrate_pronunciation.py [--db PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

# Import the same map the scorer uses, so this script can never drift from it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.metrics import PRONUNCIATION_EQUIV  # noqa: E402


def _checkpoint(conn: sqlite3.Connection) -> None:
    """Fold the WAL back into the main database file.

    Without this the updates live only in the -wal sidecar. SQLite reads the two
    together so the data looks right, but anyone who copies `stt-runs.db` on its
    own (a backup script, `docker cp`) silently gets the pre-migration rows.
    """
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.commit()


def spoken_form(match: re.Match[str], replacement: str) -> str:
    """Apply `replacement` while keeping the original word's capitalisation."""
    word = match.group(0)
    if word.isupper():
        return replacement.upper()
    if word[0].isupper():
        return replacement.capitalize()
    return replacement


def rewrite(text: str) -> str:
    """Apply every equivalence to whole words in `text`."""
    for orthographic, spoken in PRONUNCIATION_EQUIV.items():
        pattern = re.compile(rf"\b{re.escape(orthographic)}\b", re.IGNORECASE)
        text = pattern.sub(lambda m, s=spoken: spoken_form(m, s), text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(settings.db_path), help="path to stt-runs.db")
    parser.add_argument(
        "--dry-run", action="store_true", help="show the changes without writing"
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_file():
        print(f"no database at {db_path}", file=sys.stderr)
        return 1

    print(f"database: {db_path}")
    print(f"map     : {PRONUNCIATION_EQUIV}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id, expected_text FROM runs WHERE expected_text IS NOT NULL"
    ).fetchall()
    changes = [
        (row["id"], row["expected_text"], rewrite(row["expected_text"]))
        for row in rows
        if rewrite(row["expected_text"]) != row["expected_text"]
    ]

    if not changes:
        print("nothing to do — ground truth already uses the spoken form")
        if not args.dry_run:
            _checkpoint(conn)
        conn.close()
        return 0

    for run_id, before, after in changes:
        print(f"  #{run_id:<4} {before!r}\n        -> {after!r}")

    if args.dry_run:
        print(f"\ndry run: {len(changes)} row(s) would change")
        conn.close()
        return 0

    # Checkpoint first so the backup is a complete copy, not a stale snapshot.
    _checkpoint(conn)
    backup = db_path.with_suffix(f"{db_path.suffix}.{time.strftime('%Y%m%d%H%M%S')}.bak")
    shutil.copy2(db_path, backup)
    print(f"\nbackup: {backup}")

    conn.executemany(
        "UPDATE runs SET expected_text = ? WHERE id = ?",
        [(after, run_id) for run_id, _, after in changes],
    )
    conn.commit()
    _checkpoint(conn)
    conn.close()

    print(f"updated {len(changes)} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
