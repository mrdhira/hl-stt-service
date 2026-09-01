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
import sqlite3
import sys
import time
from pathlib import Path

# Import the same map the scorer uses, so this script can never drift from it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.metrics import _WORD, PRONUNCIATION_EQUIV, normalise_for_compare  # noqa: E402


def _checkpoint(conn: sqlite3.Connection) -> bool:
    """Fold the WAL back into the main database file. True if it fully ran.

    Without this the updates live only in the -wal sidecar. SQLite reads the two
    together so the data looks right, but anyone who copies `stt-runs.db` on its
    own (a backup script, `docker cp`) silently gets the pre-migration rows.

    The result is NOT ignored: `wal_checkpoint` returns `(busy, log, checkpointed)`
    and reports `busy=1` when another connection holds a read lock — which is
    the normal case when the container is running. Treating that as success is
    exactly how the stale-copy problem survives the fix meant to prevent it.
    """
    conn.commit()
    row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    # (busy, log_frames, checkpointed_frames); busy != 0 means it gave up.
    return bool(row) and row[0] == 0


def spoken_form(match: re.Match[str], replacement: str) -> str:
    """Apply `replacement` while keeping the original word's capitalisation."""
    word = match.group(0)
    if word.isupper():
        return replacement.upper()
    if word[0].isupper():
        return replacement.capitalize()
    return replacement


def rewrite(text: str) -> str:
    """Apply every equivalence to whole words in `text`, preserving case.

    Uses the scorer's own `_WORD` class rather than `\b`, which would treat
    digits and underscores as word characters and disagree with it: `\b` leaves
    "Dhira2" and "dhira_x" alone while the scorer rewrites both. Matching the
    scorer means the stored ground truth and the scored tokens agree.

    Unlike the scorer this keeps the surrounding text as written — this output
    is read by humans in the Reports table, not just compared.
    """

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        spoken = PRONUNCIATION_EQUIV.get(normalise_for_compare(word))
        return word if spoken is None else spoken_form(match, spoken)

    return _WORD.sub(replace, text)


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
    changes = []
    for row in rows:
        after = rewrite(row["expected_text"])
        if after != row["expected_text"]:
            changes.append((row["id"], row["expected_text"], after))

    if not changes:
        # Deliberately no checkpoint here: "nothing to do" must not write.
        print("nothing to do — ground truth already uses the spoken form")
        conn.close()
        return 0

    for run_id, before, after in changes:
        print(f"  #{run_id:<4} {before!r}\n        -> {after!r}")

    if args.dry_run:
        print(f"\ndry run: {len(changes)} row(s) would change")
        conn.close()
        return 0

    # Back up with SQLite's own backup API rather than copying the file. It
    # snapshots the database *including* anything sitting in the WAL, so the
    # backup is consistent even while the container holds a read lock — the case
    # where a plain shutil.copy2 of the main file silently produces a stale copy.
    backup = db_path.with_suffix(f"{db_path.suffix}.{time.strftime('%Y%m%d%H%M%S')}.bak")
    with sqlite3.connect(str(backup)) as target:
        conn.backup(target)
    print(f"\nbackup: {backup}")

    conn.executemany(
        "UPDATE runs SET expected_text = ? WHERE id = ?",
        [(after, run_id) for run_id, _, after in changes],
    )
    conn.commit()
    checkpointed = _checkpoint(conn)
    conn.close()

    print(f"updated {len(changes)} row(s)")
    if not checkpointed:
        print(
            "\nWARNING: could not checkpoint the WAL — another connection holds a\n"
            "read lock (the container is probably running). The data IS committed\n"
            "and correct when read through SQLite, but the -wal sidecar still holds\n"
            "part of it: copying stt-runs.db on its own would produce PRE-migration\n"
            "rows. Either copy stt-runs.db-wal and -shm alongside it, or stop the\n"
            "service and re-run this script to fold the WAL back in.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
