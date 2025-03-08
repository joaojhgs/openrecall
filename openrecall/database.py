import sqlite3
from collections import namedtuple
from typing import Any, List

import numpy as np

from modules.openrecall.openrecall.config import db_path
from modules.openrecall.openrecall.nlp import cosine_similarity, get_embedding

Entry = namedtuple("Entry", ["id", "app", "title", "text", "timestamp", "embedding"])


def create_db() -> None:
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS entries
               (id INTEGER PRIMARY KEY AUTOINCREMENT, app TEXT, title TEXT, text TEXT, timestamp INTEGER, embedding BLOB)"""
        )
        conn.commit()


def get_all_entries() -> List[Entry]:
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        results = c.execute("SELECT * FROM entries").fetchall()
        return [Entry(*result) for result in results]


def get_timestamps() -> List[int]:
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        results = c.execute(
            "SELECT timestamp FROM entries ORDER BY timestamp DESC"
        ).fetchall()
        return [result[0] for result in results]


def insert_entry(
    text: str, timestamp: int, embedding: Any, app: str, title: str
) -> None:
    embedding_bytes = embedding.tobytes()
    try:

        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO entries (text, timestamp, embedding, app, title) VALUES (?, ?, ?, ?, ?)",
                (text, timestamp, embedding_bytes, app, title),
            )
            conn.commit()
    except sqlite3.OperationalError as e:
        print("Error inserting entry:", e)


def search_entries(query: str, limit: int = 10) -> List[Entry]:
    entries = get_all_entries()
    embeddings = [np.frombuffer(entry.embedding, dtype=np.float64) for entry in entries]
    query_embedding = get_embedding(query)
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    indices = np.argsort(similarities)[::-1]
    sorted_entries = [entries[i] for i in indices]
    return sorted_entries[:limit]
