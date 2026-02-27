"""
Minimal runnable example for the Multimodal Coordination Detection Toolkit.

Demonstrates the end-to-end API using the bundled toy dataset, which already
contains precomputed CLIP embeddings — no GPU or image downloads required.

Usage
-----
    uv run python run_example.py
    # or, with the venv activated:
    python run_example.py

Expected output
---------------
    TiCNet.graphml and LiCNet.graphml in ./results/
"""

import os
import re
import sys
from ast import literal_eval

import numpy as np
import pandas as pd

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multimodal_coordination import detect_coordination

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "toy_dataset_for_test.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _fix_array_string(s: str) -> str:
    """Insert missing commas in numpy array string representations."""
    inner = s.strip()[1:-1]
    inner = re.sub(r"\s+", " ", inner)
    inner = re.sub(r"(?<=[0-9e]) (?=[-]?[0-9])", ", ", inner)
    return f"[{inner}]"


def load_toy_dataset(csv_path: str) -> pd.DataFrame:
    print(f"Loading toy dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    print("Parsing precomputed embeddings from string format...")
    df["text_emb_from_multi"] = (
        df["text_emb_from_multi"]
        .apply(_fix_array_string)
        .apply(literal_eval)
        .apply(np.array)
    )
    df["image_embed"] = (
        df["image_embed"]
        .apply(_fix_array_string)
        .apply(literal_eval)
        .apply(np.array)
    )
    return df


def run():
    df = load_toy_dataset(CSV_PATH)

    print("\nRunning coordination detection pipeline...")
    results = detect_coordination(
        posts_df=df,
        output_dir=OUTPUT_DIR,

        # The toy dataset has precomputed embeddings — skip the CLIP step
        precomputed_embeddings=True,
        text_embed_col="text_emb_from_multi",
        image_embed_col="image_embed",

        # Column mappings matching the toy dataset CSV
        message_id_col="id",
        user_id_col="author_id",
        username_col="username",
        timestamp_col="timestamp",
        repost_id_col="retweet_id",
        reply_id_col="reply_id",
        url_col="url",

        # Both network types
        network_type="both",

        # Parameters tuned for the small toy dataset
        time_window=3600,       # 1-hour window
        text_threshold=0.9,
        img_threshold=0.8,
        min_edge_weight=1,
        min_community_size=2,

        show_progress_bar=True,
    )

    print("\n" + "=" * 50)
    print("Results:")
    for name, path in results.items():
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  {name}: {path}  ({size_kb:.1f} KB)")
        else:
            print(f"  {name}: {path}  (file not found — network may be empty)")

    print("\nTo visualise: open the .graphml files in Gephi or Cytoscape.")
    return results


# The if __name__ == '__main__' guard is required on macOS/Windows because
# multiprocessing uses the 'spawn' start method and will reimport this module
# in child processes — without the guard, child processes would try to run
# detect_coordination again, causing a recursive crash.
if __name__ == "__main__":
    run()
