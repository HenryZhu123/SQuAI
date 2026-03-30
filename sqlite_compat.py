"""
Key-value store compatibility layer: plyvel.LevelDB when available, else SQLite.

SQuAI uses LevelDB (plyvel) for full-text blobs. On Windows, plyvel often fails to
build; this module provides SQLiteDB with the same get/put/close/iteration surface
so callers can use open_db() without branching.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Iterator, Optional, Tuple, Union

# Prefer native LevelDB when the extension is installed (Linux / many CI images).
try:
    import plyvel  # type: ignore

    HAS_PLYVEL = True
except ImportError:
    plyvel = None  # type: ignore
    HAS_PLYVEL = False

__all__ = ["open_db", "SQLiteDB", "HAS_PLYVEL"]


class SQLiteDB:
    """
    Minimal LevelDB-like API backed by a single SQLite file under a directory.

    plyvel stores data in a *directory*; we mirror that by placing
    ``_sqkv.sqlite`` inside that directory so existing DB_PATH values stay valid.
    """

    _TABLE = "kv"

    def __init__(self, path: str, create_if_missing: bool = True) -> None:
        self._dir = os.path.abspath(path)
        self._dbfile = os.path.join(self._dir, "_sqkv.sqlite")
        self._conn: Optional[sqlite3.Connection] = None

        if not create_if_missing:
            if not os.path.isdir(self._dir):
                raise OSError(f"Database directory does not exist: {self._dir}")
            if not os.path.isfile(self._dbfile):
                raise OSError(f"SQLite KV store not found: {self._dbfile}")
        else:
            os.makedirs(self._dir, exist_ok=True)

        # check_same_thread=False matches typical plyvel usage from FastAPI / threads
        self._conn = sqlite3.connect(self._dbfile, check_same_thread=False)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._TABLE} (
                key BLOB PRIMARY KEY,
                value BLOB NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, key: bytes, default: Optional[bytes] = None) -> Optional[bytes]:
        """Return value for key, or default / None if missing (LevelDB-like)."""
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes")
        cur = self._conn.execute(
            f"SELECT value FROM {self._TABLE} WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        if row is None:
            return default
        return row[0]

    def put(self, key: bytes, value: bytes, sync: bool = False) -> None:
        """Insert or replace key. ``sync`` is accepted for API parity (ignored)."""
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes")
        if not isinstance(value, bytes):
            raise TypeError("value must be bytes")
        self._conn.execute(
            f"INSERT OR REPLACE INTO {self._TABLE} (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()
        if sync:
            pass  # SQLite commits synchronously by default in this simple path

    def delete(self, key: bytes) -> None:
        """Remove key if present (extension for parity with common KV stores)."""
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes")
        self._conn.execute(f"DELETE FROM {self._TABLE} WHERE key = ?", (key,))
        self._conn.commit()

    def __iter__(self) -> Iterator[Tuple[bytes, bytes]]:
        """Yield (key, value) pairs in key order (deterministic iteration)."""
        cur = self._conn.execute(
            f"SELECT key, value FROM {self._TABLE} ORDER BY key"
        )
        for key, value in cur:
            yield key, value

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SQLiteDB":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def open_db(
    path: Union[str, os.PathLike[str]],
    create_if_missing: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Open a LevelDB-backed database when plyvel is available; otherwise SQLiteDB.

    Parameters mirror plyvel.DB(path, create_if_missing=..., **kwargs) for the
    native path; extra kwargs are forwarded only to plyvel.

    Returns:
        plyvel.DB or SQLiteDB instance.
    """
    path_str = os.path.abspath(os.fspath(path))

    if HAS_PLYVEL:
        return plyvel.DB(path_str, create_if_missing=create_if_missing, **kwargs)

    if kwargs:
        # SQLite backend does not support bloom filters, caches, etc.
        import warnings

        warnings.warn(
            f"SQLiteDB ignores plyvel-specific kwargs: {list(kwargs.keys())}",
            stacklevel=2,
        )

    return SQLiteDB(path_str, create_if_missing=create_if_missing)
