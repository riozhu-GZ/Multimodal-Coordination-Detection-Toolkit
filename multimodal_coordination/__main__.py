"""
Command-line interface for the Multimodal Coordination Detection Toolkit.

Usage
-----
    python -m multimodal_coordination --help

    # Quickest path (precomputed embeddings already in CSV)
    python -m multimodal_coordination \\
        --input dataset/toy_dataset_for_test.csv \\
        --output-dir ./results \\
        --precomputed-embeddings \\
        --network-type both

    # Full pipeline (raw images + text → embeddings → networks)
    python -m multimodal_coordination \\
        --input posts.csv \\
        --output-dir ./results \\
        --images-dir ./images \\
        --text-col text \\
        --image-col image_filename \\
        --network-type TiCNet
"""

import argparse
import sys

import pandas as pd

from .config import (
    DEFAULT_IMG_THRESHOLD,
    DEFAULT_MIN_COMMUNITY_SIZE,
    DEFAULT_MIN_EDGE_WEIGHT,
    DEFAULT_N_THREADS,
    DEFAULT_TEXT_THRESHOLD,
    DEFAULT_TIME_WINDOW,
    NETWORK_BOTH,
    SIM_MULTIMODAL_DISJOINT,
    VALID_NETWORK_TYPES,
    VALID_SIM_TYPES,
)
from .pipeline import detect_coordination


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m multimodal_coordination",
        description=(
            "Multimodal Coordination Detection Toolkit\n\n"
            "Detects coordinated inauthentic behaviour by finding social media\n"
            "accounts that post similar text+image content within a time window.\n"
            "Outputs TiCNet and/or LiCNet coordination networks as GraphML files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- I/O ---
    io = p.add_argument_group("Input / Output")
    io.add_argument(
        "--input", "-i", required=True, metavar="CSV",
        help="Path to the input CSV file containing post data.",
    )
    io.add_argument(
        "--output-dir", "-o", default="./results", metavar="DIR",
        help="Directory where GraphML output files will be saved (default: ./results).",
    )
    io.add_argument(
        "--db-path", default=None, metavar="DB",
        help=(
            "Path for the intermediate SQLite database. "
            "If omitted, a temporary file is created and removed automatically."
        ),
    )

    # --- Embedding control ---
    emb = p.add_argument_group("Embedding options")
    emb.add_argument(
        "--precomputed-embeddings", action="store_true",
        help=(
            "Skip CLIP embedding generation. "
            "The CSV must already contain --text-embed-col and --image-embed-col columns."
        ),
    )
    emb.add_argument(
        "--images-dir", default=None, metavar="DIR",
        help=(
            "Base directory for image files (only needed when --precomputed-embeddings is NOT set). "
            "Values in --image-col are joined to this directory."
        ),
    )
    emb.add_argument(
        "--text-col", default="text", metavar="COL",
        help="Column with raw post text (default: 'text').",
    )
    emb.add_argument(
        "--image-col", default="image_path", metavar="COL",
        help="Column with image filename/path (default: 'image_path').",
    )
    emb.add_argument(
        "--text-embed-col", default="text_emb_from_multi", metavar="COL",
        help="Column with precomputed text embeddings (default: 'text_emb_from_multi').",
    )
    emb.add_argument(
        "--image-embed-col", default="image_embed", metavar="COL",
        help="Column with precomputed image embeddings (default: 'image_embed').",
    )

    # --- Column mappings ---
    cols = p.add_argument_group("Column name mappings")
    cols.add_argument("--user-id-col", default="author_id", metavar="COL",
                      help="Account ID column (default: 'author_id').")
    cols.add_argument("--username-col", default="username", metavar="COL",
                      help="Account username column (default: 'username').")
    cols.add_argument("--message-id-col", default="id", metavar="COL",
                      help="Post ID column (default: 'id').")
    cols.add_argument("--timestamp-col", default="timestamp", metavar="COL",
                      help="Unix timestamp column (default: 'timestamp').")
    cols.add_argument("--repost-id-col", default="retweet_id", metavar="COL",
                      help="Repost/retweet parent ID column (default: 'retweet_id').")
    cols.add_argument("--reply-id-col", default="reply_id", metavar="COL",
                      help="Reply parent ID column (default: 'reply_id').")
    cols.add_argument("--url-col", default="url", metavar="COL",
                      help="URL column (default: 'url').")

    # --- Network parameters ---
    net = p.add_argument_group("Network detection parameters")
    net.add_argument(
        "--network-type", default=NETWORK_BOTH,
        choices=VALID_NETWORK_TYPES,
        help=f"Which network(s) to compute (default: '{NETWORK_BOTH}').",
    )
    net.add_argument(
        "--time-window", type=int, default=DEFAULT_TIME_WINDOW, metavar="SECONDS",
        help=f"Max time gap between similar posts in seconds (default: {DEFAULT_TIME_WINDOW}).",
    )
    net.add_argument(
        "--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD, metavar="FLOAT",
        help=f"Cosine similarity threshold for text (default: {DEFAULT_TEXT_THRESHOLD}).",
    )
    net.add_argument(
        "--img-threshold", type=float, default=DEFAULT_IMG_THRESHOLD, metavar="FLOAT",
        help=f"Cosine similarity threshold for images (default: {DEFAULT_IMG_THRESHOLD}).",
    )
    net.add_argument(
        "--min-edge-weight", type=int, default=DEFAULT_MIN_EDGE_WEIGHT, metavar="INT",
        help=f"Minimum co-similar post pairs to form an edge (default: {DEFAULT_MIN_EDGE_WEIGHT}).",
    )
    net.add_argument(
        "--min-community-size", type=int, default=DEFAULT_MIN_COMMUNITY_SIZE, metavar="INT",
        help=f"Minimum Leiden community size for TiCNet (default: {DEFAULT_MIN_COMMUNITY_SIZE}).",
    )
    net.add_argument(
        "--measure-type", default=SIM_MULTIMODAL_DISJOINT,
        choices=VALID_SIM_TYPES,
        help=f"Similarity measurement strategy (default: '{SIM_MULTIMODAL_DISJOINT}').",
    )
    net.add_argument(
        "--n-threads", type=int, default=DEFAULT_N_THREADS, metavar="INT",
        help=f"Number of parallel worker threads (default: {DEFAULT_N_THREADS}).",
    )
    net.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output.",
    )

    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    print(f"Loading input data from: {args.input}")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read input file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    try:
        results = detect_coordination(
            posts_df=df,
            output_dir=args.output_dir,
            precomputed_embeddings=args.precomputed_embeddings,
            images_dir=args.images_dir,
            text_col=args.text_col,
            image_col=args.image_col,
            text_embed_col=args.text_embed_col,
            image_embed_col=args.image_embed_col,
            user_id_col=args.user_id_col,
            username_col=args.username_col,
            message_id_col=args.message_id_col,
            timestamp_col=args.timestamp_col,
            repost_id_col=args.repost_id_col,
            reply_id_col=args.reply_id_col,
            url_col=args.url_col,
            network_type=args.network_type,
            time_window=args.time_window,
            text_threshold=args.text_threshold,
            img_threshold=args.img_threshold,
            min_edge_weight=args.min_edge_weight,
            min_community_size=args.min_community_size,
            measure_type=args.measure_type,
            n_threads=args.n_threads,
            db_path=args.db_path,
            show_progress_bar=not args.quiet,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nResults:")
    for name, path in results.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
