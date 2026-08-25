"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.base import BaseCheckpointSaver


def build_checkpointer(
    kind: str = "memory", database_url: str | None = None
) -> BaseCheckpointSaver | None:
    """Return a LangGraph checkpointer.

    - "none": no persistence
    - "memory": in-process MemorySaver (resets on restart)
    - "sqlite": SqliteSaver backed by a local file (survives process restarts)
    - "postgres": PostgresSaver (requires a running database)
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = database_url or "checkpoints.db"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return SqliteSaver(conn=conn)
    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import (  # type: ignore[import-not-found]
                PostgresSaver,
            )
        except ImportError as exc:
            raise RuntimeError("Install: pip install langgraph-checkpoint-postgres") from exc
        if not database_url:
            raise ValueError("database_url is required for the postgres checkpointer")
        return PostgresSaver.from_conn_string(database_url)
    raise ValueError(f"Unknown checkpointer kind: {kind}")
