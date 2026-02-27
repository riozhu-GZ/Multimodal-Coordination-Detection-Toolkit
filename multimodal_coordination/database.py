"""
Database initialisation and data ingestion for the Multimodal Coordination
Detection Toolkit.

Manages a SQLite database that stores posts with their text and image
embeddings, which is then queried by the coordination detection algorithms.
"""

import concurrent.futures
import os
import re
import sqlite3 as lite
from ast import literal_eval
from typing import Callable, Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Table name mappings (used by detection and graph-loading utilities)
# ---------------------------------------------------------------------------

COMMAND_TABLE = {
    "co_retweet": "co_retweet_network",
    "co_tweet": "co_tweet_network",
    "co_reply": "co_reply_network",
    "co_link": "co_link_network",
    "co_similar_tweet": "co_similar_tweet_network",
    "co_post": "co_post_network",
}

COMMAND_MULTIMODAL_TABLE = {
    "co_similar_multimodal": "co_similar_multimodal_network",
    "co_similar_image": "co_similar_image",
}


# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------

def initialise_multimodal_db(db_path: str) -> lite.Connection:
    """
    Initialise the SQLite database with the required schema.

    Creates tables and indexes if they do not already exist. Raises a
    ValueError if the on-disk format is incompatible with this version.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (created if it does not exist).

    Returns
    -------
    sqlite3.Connection
        Open connection to the initialised database.
    """
    db = lite.connect(db_path, isolation_level=None)

    db.executescript(
        """
        pragma journal_mode=WAL;
        pragma synchronous=normal;

        create table if not exists edge (
            message_id primary key,
            user_id not null,
            username text,
            repost_id,
            reply_id,
            text_embed,
            image_embed,
            transformed_message text,
            transformed_message_length integer,
            transformed_message_hash blob,
            token_set text,
            timestamp integer
        );

        create index if not exists user_edge on edge(user_id);

        create table if not exists message_url(
            message_id references edge(message_id),
            url,
            timestamp,
            user_id,
            primary key (message_id, url)
        );

        create table if not exists resolved_url(
            url primary key,
            resolved_url,
            ssl_verified,
            resolved_status
        );

        create trigger if not exists url_to_resolve after insert on message_url
            begin
                insert or ignore into resolved_url(url) values(new.url);
            end;

        create table if not exists metadata (
            property primary key,
            value
        );

        insert or ignore into metadata values('version', 1);
        """
    )

    edge_columns = {row[1] for row in db.execute("pragma table_info('edge')")}
    version = list(db.execute("select value from metadata where property = 'version'"))[0][0]

    if "message_length" in edge_columns or version != 1:
        raise ValueError(
            "This database is not compatible with this version of the "
            "coordination network toolkit — you will need to reprocess your "
            "data into a new database."
        )

    return db


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

def preprocess_multimodal_data(db_path: str, messages: Iterable) -> None:
    """
    Insert a batch of messages (with optional embeddings) into the database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    messages : iterable of tuples
        Each tuple: (message_id, user_id, username, repost_id, reply_id,
                     text_embed, image_embed, timestamp, urls)
        - ``text_embed`` / ``image_embed``: comma-separated float strings or None.
        - ``urls``: list of URL strings (ignored for reposts).
    """
    db = lite.connect(db_path, isolation_level=None)

    try:
        db.execute("begin")
        for row in messages:
            (
                message_id, user_id, username,
                repost_id, reply_id,
                text_embed, image_embed,
                timestamp, urls,
            ) = row

            db.execute(
                "insert or ignore into edge values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    user_id,
                    username,
                    repost_id or None,
                    reply_id or None,
                    text_embed or None,
                    image_embed or None,
                    None,   # transformed_message (populated later if needed)
                    None,   # transformed_message_length
                    None,   # transformed_message_hash
                    None,   # token_set
                    float(timestamp),
                ),
            )

            # Record URLs (only for original posts, not reposts)
            if not repost_id:
                for url in (urls or []):
                    db.execute(
                        "insert or ignore into message_url values(?, ?, ?, ?)",
                        (message_id, url, float(timestamp), user_id),
                    )

        db.execute("commit")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helper: embedding serialisation
# ---------------------------------------------------------------------------

def _embedding_to_str(emb) -> Optional[str]:
    """Convert a list/array embedding to a comma-separated string, or None."""
    if emb is None:
        return None
    if isinstance(emb, np.ndarray):
        emb = emb.tolist()
    if isinstance(emb, list):
        return ",".join(map(str, emb))
    return str(emb)


def _fix_array_string(array_str: str) -> str:
    """Fix missing commas in numpy array string representations."""
    inner = array_str.strip()[1:-1]
    inner = re.sub(r"\s+", " ", inner)
    inner = re.sub(r"(?<=[0-9e]) (?=[-]?[0-9])", ", ", inner)
    return f"[{inner}]"


# ---------------------------------------------------------------------------
# DataFrame loading and chunked ingestion
# ---------------------------------------------------------------------------

def load_and_prepare_dataframe(
    csv_path: str,
    text_embed_col: str = "text_emb_from_multi",
    image_embed_col: str = "image_embed",
) -> pd.DataFrame:
    """
    Load a CSV dataset and parse string-encoded embeddings to numpy arrays.

    Parameters
    ----------
    csv_path : str
        Path to CSV file.
    text_embed_col : str
        Column name containing text embeddings (as numpy-style strings).
    image_embed_col : str
        Column name containing image embeddings.

    Returns
    -------
    pd.DataFrame
        DataFrame with embedding columns converted to numpy arrays.
    """
    df = pd.read_csv(csv_path)
    df[text_embed_col] = (
        df[text_embed_col]
        .apply(_fix_array_string)
        .apply(literal_eval)
        .apply(np.array)
    )
    df[image_embed_col] = (
        df[image_embed_col]
        .apply(_fix_array_string)
        .apply(literal_eval)
        .apply(np.array)
    )
    return df


def divide_dataframe_into_chunks(df: pd.DataFrame, window_size: int) -> List:
    """
    Split a DataFrame into sequential, non-overlapping time-ordered chunks.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'timestamp' column.
    window_size : int
        Maximum number of rows per chunk. If the DataFrame is smaller than
        ``window_size``, a single chunk containing all rows is returned.

    Returns
    -------
    list of pd.Index
        List of DataFrame index arrays, each representing one chunk.
    """
    df = df.sort_values(by="timestamp")
    effective_window = min(window_size, len(df))
    chunks = [
        df.iloc[i : i + effective_window].index
        for i in range(0, len(df), effective_window)
    ]
    return chunks


def _ingest_chunk(
    index: int,
    chunk: pd.DataFrame,
    db_path: str,
    text_embed_col: str = "text_emb_from_multi",
    image_embed_col: str = "image_embed",
    url_col: str = "url",
    message_id_col: str = "id",
    user_id_col: str = "author_id",
    username_col: str = "username",
    repost_id_col: str = "retweet_id",
    reply_id_col: str = "reply_id",
    timestamp_col: str = "timestamp",
) -> int:
    """Ingest a single DataFrame chunk into the database. Returns row count."""
    if index % 500 == 0 and index != 0:
        print(f"Processing chunk {index}")

    initialise_multimodal_db(db_path)

    def to_str(val):
        if val is None:
            return None
        if isinstance(val, np.ndarray):
            return ",".join(map(str, val.tolist()))
        if isinstance(val, list):
            return ",".join(map(str, val))
        return str(val)

    rows = []
    for _, row in chunk.iterrows():
        url_val = row.get(url_col, "")
        urls = str(url_val).split() if pd.notna(url_val) else []
        rows.append((
            row[message_id_col],
            row[user_id_col],
            row[username_col],
            row.get(repost_id_col) if pd.notna(row.get(repost_id_col, None)) else None,
            row.get(reply_id_col) if pd.notna(row.get(reply_id_col, None)) else None,
            to_str(row[text_embed_col]),
            to_str(row[image_embed_col]),
            row[timestamp_col],
            urls,
        ))

    preprocess_multimodal_data(db_path, rows)

    conn = lite.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM edge").fetchone()[0]
    conn.close()
    return count


def generate_database_from_dataframe(
    df: pd.DataFrame,
    db_path: str,
    window_size: int = 10000,
    text_embed_col: str = "text_emb_from_multi",
    image_embed_col: str = "image_embed",
    url_col: str = "url",
    message_id_col: str = "id",
    user_id_col: str = "author_id",
    username_col: str = "username",
    repost_id_col: str = "retweet_id",
    reply_id_col: str = "reply_id",
    timestamp_col: str = "timestamp",
    n_workers: int = None,
) -> str:
    """
    Ingest an entire DataFrame into a SQLite database in parallel chunks.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with embedding columns already populated.
    db_path : str
        Target SQLite database path.
    window_size : int
        Rows per ingestion chunk.
    text_embed_col, image_embed_col : str
        Column names for precomputed embeddings.
    url_col, message_id_col, user_id_col, username_col : str
        Column name mappings.
    repost_id_col, reply_id_col, timestamp_col : str
        Column name mappings.
    n_workers : int, optional
        Number of parallel worker threads (defaults to CPU count).

    Returns
    -------
    str
        The ``db_path`` that was written.
    """
    n_workers = n_workers or os.cpu_count()
    chunk_indices = divide_dataframe_into_chunks(df, window_size=window_size)

    print(f"Ingesting {len(df)} rows in {len(chunk_indices)} chunks...")

    with tqdm(total=len(chunk_indices), desc="Ingesting chunks") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for idx, chunk_idx in enumerate(chunk_indices):
                chunk = df.loc[chunk_idx]
                future = executor.submit(
                    _ingest_chunk,
                    idx, chunk, db_path,
                    text_embed_col, image_embed_col,
                    url_col, message_id_col, user_id_col,
                    username_col, repost_id_col, reply_id_col, timestamp_col,
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                future.result()  # propagate exceptions
                pbar.update(1)

    return db_path


def check_database_size(db_path: str) -> int:
    """Return the number of rows in the edge table."""
    conn = lite.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM edge").fetchone()[0]
    conn.close()
    return count
