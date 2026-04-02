"""
PostgreSQL-backed key-value store for SQuAI full-text blobs (paper_id -> content).

Connect with a standard DSN; do not use the server's data directory path
(e.g. /var/lib/postgresql/16/main) from application code — that is the cluster
storage managed by PostgreSQL, not a client connection target.

Resolution (see config.py):
  FULLTEXT_PG_DSN         from env SQUAI_FULLTEXT_PG_DSN, else FULLTEXT_PG_DSN_DEFAULT in config.py
  FULLTEXT_PG_TABLE       from env SQUAI_FULLTEXT_PG_TABLE, else FULLTEXT_PG_TABLE_DEFAULT in config.py

Other:
  SQUAI_PG_INGEST_BATCH   batch size for buffered writes (default: 500)
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Iterator, List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_batch

_TABLE_SAFE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")


def _validate_table_name(name: str) -> str:
    if not _TABLE_SAFE.match(name):
        raise ValueError(
            f"Invalid table name {name!r}: use letters, digits, underscore; max 63 chars (PostgreSQL limit)."
        )
    return name


class PostgresKV:
    """
    LevelDB-like API: get/put/close/__iter__ over a single table (key TEXT, value BYTEA).
    Thread-safe via a lock. Puts are buffered and flushed in batches for ingest performance.
    """

    def __init__(
        self,
        dsn: str,
        table: str = "squai_fulltext_kv",
        create_if_missing: bool = True,
    ) -> None:
        self._dsn = dsn
        self._table = _validate_table_name(table)
        self._lock = threading.RLock()
        self._batch: List[Tuple[str, bytes]] = []
        self._batch_size = max(1, int(os.environ.get("SQUAI_PG_INGEST_BATCH", "500")))
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        if create_if_missing:
            self._create_table()
        else:
            self._assert_table_exists()

    def _create_table(self) -> None:
        # Table name validated to [a-zA-Z0-9_]+ only — safe for identifier interpolation.
        t = self._table
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {t} "
                    f"(key TEXT PRIMARY KEY, value BYTEA NOT NULL)"
                )
            self._conn.commit()

    def _assert_table_exists(self) -> None:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                    """,
                    (self._table,),
                )
                if cur.fetchone() is None:
                    raise OSError(
                        f"PostgreSQL table {self._table!r} does not exist "
                        "(create_if_missing=False). Run ingest with create_if_missing=True first."
                    )

    def _flush_batch(self) -> None:
        if not self._batch:
            return
        chunk = self._batch
        self._batch = []
        t = self._table
        query = (
            f"INSERT INTO {t} (key, value) VALUES (%s, %s) "
            f"ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
        )
        page = min(500, max(1, len(chunk)))
        with self._conn.cursor() as cur:
            execute_batch(cur, query, chunk, page_size=page)
        self._conn.commit()

    def get(self, key: bytes, default: Optional[bytes] = None) -> Optional[bytes]:
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes")
        skey = key.decode("utf-8")
        t = self._table
        with self._lock:
            self._flush_batch()
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT value FROM {t} WHERE key = %s", (skey,))
                row = cur.fetchone()
        if row is None:
            return default
        return bytes(row[0])

    def put(self, key: bytes, value: bytes, sync: bool = False) -> None:
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes")
        if not isinstance(value, bytes):
            raise TypeError("value must be bytes")
        skey = key.decode("utf-8")
        with self._lock:
            self._batch.append((skey, value))
            if sync or len(self._batch) >= self._batch_size:
                self._flush_batch()

    def delete(self, key: bytes) -> None:
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes")
        skey = key.decode("utf-8")
        t = self._table
        with self._lock:
            self._flush_batch()
            with self._conn.cursor() as cur:
                cur.execute(f"DELETE FROM {t} WHERE key = %s", (skey,))
            self._conn.commit()

    def __iter__(self) -> Iterator[Tuple[bytes, bytes]]:
        t = self._table
        with self._lock:
            self._flush_batch()
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT key, value FROM {t} ORDER BY key")
                rows = cur.fetchall()
        for k, v in rows:
            yield k.encode("utf-8"), bytes(v)

    def close(self) -> None:
        with self._lock:
            self._flush_batch()
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "PostgresKV":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def postgres_dsn_resolved() -> Optional[str]:
    """Effective DSN after config.py defaults and env override."""
    from config import FULLTEXT_PG_DSN

    dsn = (FULLTEXT_PG_DSN or "").strip()
    return dsn or None


def postgres_dsn_from_env() -> Optional[str]:
    """Backward-compatible alias for :func:`postgres_dsn_resolved`."""
    return postgres_dsn_resolved()


def postgres_table_from_config() -> str:
    from config import FULLTEXT_PG_TABLE

    t = (FULLTEXT_PG_TABLE or "squai_fulltext_kv").strip()
    return _validate_table_name(t)


def postgres_table_from_env() -> str:
    """Backward-compatible alias for :func:`postgres_table_from_config`."""
    return postgres_table_from_config()
