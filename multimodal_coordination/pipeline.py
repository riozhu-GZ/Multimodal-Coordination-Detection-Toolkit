"""
End-to-end coordination detection pipeline.

Provides the single public entry point :func:`detect_coordination`, which
accepts a DataFrame of social media posts (with or without precomputed
embeddings) and produces GraphML coordination network files.
"""

import os
import tempfile
from typing import Optional

import pandas as pd

from .config import (
    DEFAULT_IMG_THRESHOLD,
    DEFAULT_MIN_COMMUNITY_SIZE,
    DEFAULT_MIN_EDGE_WEIGHT,
    DEFAULT_N_THREADS,
    DEFAULT_TEXT_THRESHOLD,
    DEFAULT_TIME_WINDOW,
    LICNET_MIN_EDGE_WEIGHT,
    LICNET_TIME_WINDOW,
    NETWORK_BOTH,
    NETWORK_LICNET,
    NETWORK_TICNET,
    SIM_MULTIMODAL_DISJOINT,
    VALID_NETWORK_TYPES,
)
from .database import generate_database_from_dataframe, initialise_multimodal_db
from .networks import compute_licnet, compute_ticnet


def detect_coordination(
    posts_df: pd.DataFrame,
    output_dir: str,
    # --- Embedding control ---
    precomputed_embeddings: bool = False,
    images_dir: Optional[str] = None,
    text_col: str = "text",
    image_col: str = "image_path",
    text_embed_col: str = "text_emb_from_multi",
    image_embed_col: str = "image_embed",
    # --- Column mappings ---
    user_id_col: str = "author_id",
    username_col: str = "username",
    message_id_col: str = "id",
    timestamp_col: str = "timestamp",
    repost_id_col: str = "retweet_id",
    reply_id_col: str = "reply_id",
    url_col: str = "url",
    # --- Network parameters ---
    network_type: str = NETWORK_BOTH,
    time_window: int = DEFAULT_TIME_WINDOW,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    img_threshold: float = DEFAULT_IMG_THRESHOLD,
    min_edge_weight: int = DEFAULT_MIN_EDGE_WEIGHT,
    min_community_size: int = DEFAULT_MIN_COMMUNITY_SIZE,
    measure_type: str = SIM_MULTIMODAL_DISJOINT,
    n_threads: int = DEFAULT_N_THREADS,
    # --- Internal ---
    db_path: Optional[str] = None,
    ingestion_window_size: int = 10000,
    show_progress_bar: bool = True,
) -> dict:
    """
    End-to-end multimodal coordination detection pipeline.

    Takes a DataFrame of social media posts and produces GraphML files
    representing coordination networks between accounts.

    Parameters
    ----------
    posts_df : pd.DataFrame
        Input DataFrame. Required columns depend on ``precomputed_embeddings``:

        - Always required: ``message_id_col``, ``user_id_col``, ``username_col``,
          ``timestamp_col``
        - If ``precomputed_embeddings=True``: ``text_embed_col``, ``image_embed_col``
          (comma-separated float strings or numpy arrays)
        - If ``precomputed_embeddings=False``: ``text_col``, ``image_col``
          (raw text and image filename/path)

    output_dir : str
        Directory where GraphML files will be saved. Created if it does not exist.

    precomputed_embeddings : bool
        If True, ``text_embed_col`` and ``image_embed_col`` must already exist
        in ``posts_df``. The embedding step is skipped.
        If False, CLIP embeddings are generated from raw text and images.

    images_dir : str, optional
        Base directory for image files. Used when ``precomputed_embeddings=False``
        and ``image_col`` contains relative filenames rather than absolute paths.

    text_col : str
        Column with raw post text (used when ``precomputed_embeddings=False``).

    image_col : str
        Column with image filename or path (used when ``precomputed_embeddings=False``).

    text_embed_col : str
        Column with precomputed text embeddings (used when ``precomputed_embeddings=True``).

    image_embed_col : str
        Column with precomputed image embeddings (used when ``precomputed_embeddings=True``).

    user_id_col : str
        Column with numeric/string account identifiers.

    username_col : str
        Column with human-readable account names.

    message_id_col : str
        Column with unique post identifiers.

    timestamp_col : str
        Column with Unix timestamps (integer seconds).

    repost_id_col : str
        Column with repost/retweet parent ID (None/NaN for original posts).

    reply_id_col : str
        Column with reply parent ID (None/NaN for non-replies).

    url_col : str
        Column with space-separated URLs included in the post.

    network_type : str
        Which network(s) to compute:
        - ``'TiCNet'``: Tweet-image Coordination Network (strict, community-filtered)
        - ``'LiCNet'``: Loose image Coordination Network (direct edge weights)
        - ``'both'``: compute both

    time_window : int
        Maximum time gap in seconds between similar posts to count as coordinated.
        Used for TiCNet (LiCNet uses a separate larger window by default).

    text_threshold : float
        Cosine similarity threshold for text (0–1). Default: 0.9.

    img_threshold : float
        Cosine similarity threshold for images (0–1). Default: 0.8.

    min_edge_weight : int
        Minimum number of co-similar post pairs required to form an edge.
        Used for TiCNet. LiCNet uses a higher default (5).

    min_community_size : int
        Minimum community size retained after Leiden community detection (TiCNet only).

    measure_type : str
        Similarity measurement strategy. One of:
        ``'multimodal_disjoint'`` (text OR image), ``'text_only'``,
        ``'image_only'``, ``'multimodal_joint'`` (concatenated).

    n_threads : int
        Number of parallel worker processes for the detection step.

    db_path : str, optional
        Path for the SQLite database. If None, a temporary file is created
        and deleted after the pipeline completes.

    ingestion_window_size : int
        Number of rows per database ingestion chunk.

    show_progress_bar : bool
        Print progress information to stdout.

    Returns
    -------
    dict
        Keys are ``'TiCNet'`` and/or ``'LiCNet'`` depending on ``network_type``.
        Values are absolute paths to the corresponding ``.graphml`` files.

    Examples
    --------
    >>> import pandas as pd
    >>> from multimodal_coordination import detect_coordination
    >>>
    >>> df = pd.read_csv("dataset/toy_dataset_for_test.csv")
    >>> results = detect_coordination(
    ...     df,
    ...     output_dir="./results",
    ...     precomputed_embeddings=True,
    ...     network_type="both",
    ... )
    >>> print(results)
    {'TiCNet': '/abs/path/results/TiCNet.graphml', 'LiCNet': '/abs/path/results/LiCNet.graphml'}
    """
    # --- Validate inputs ---
    if network_type not in VALID_NETWORK_TYPES:
        raise ValueError(
            f"Invalid network_type: '{network_type}'. "
            f"Must be one of {VALID_NETWORK_TYPES}."
        )

    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Embedding ---
    df = posts_df.copy()

    if not precomputed_embeddings:
        from .embeddings import embed_dataframe, load_clip_model
        if show_progress_bar:
            print("Step 1/4: Generating CLIP embeddings...")
        model = load_clip_model()
        df = embed_dataframe(
            df,
            text_col=text_col,
            image_col=image_col,
            images_dir=images_dir,
            model=model,
        )
        # Rename generated columns to internal names
        text_embed_col = "text_embed"
        image_embed_col = "image_embed"
    else:
        if show_progress_bar:
            print("Step 1/4: Using precomputed embeddings (skipping CLIP).")
        for col in [text_embed_col, image_embed_col]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in posts_df. "
                    f"Set precomputed_embeddings=False or provide the correct column name."
                )

    # --- Step 2: Database ingestion ---
    _tmp_db = None
    if db_path is None:
        _tmp_fd, db_path = tempfile.mkstemp(suffix=".db", prefix="multimodal_coord_")
        os.close(_tmp_fd)
        _tmp_db = db_path

    if show_progress_bar:
        print(f"Step 2/4: Ingesting {len(df)} posts into database: {db_path}")

    initialise_multimodal_db(db_path)
    generate_database_from_dataframe(
        df,
        db_path=db_path,
        window_size=ingestion_window_size,
        text_embed_col=text_embed_col,
        image_embed_col=image_embed_col,
        url_col=url_col,
        message_id_col=message_id_col,
        user_id_col=user_id_col,
        username_col=username_col,
        repost_id_col=repost_id_col,
        reply_id_col=reply_id_col,
        timestamp_col=timestamp_col,
        n_workers=n_threads,
    )

    # --- Step 3 & 4: Network detection and export ---
    if show_progress_bar:
        print(f"Step 3/4: Detecting coordination (network_type={network_type})...")

    results = {}

    try:
        if network_type in (NETWORK_TICNET, NETWORK_BOTH):
            ticnet_path = os.path.abspath(os.path.join(output_dir, "TiCNet.graphml"))
            if show_progress_bar:
                print(f"  → Computing TiCNet → {ticnet_path}")
            compute_ticnet(
                db_path=db_path,
                tweet_df=df,
                output_path=ticnet_path,
                time_window=time_window,
                text_threshold=text_threshold,
                img_threshold=img_threshold,
                min_edge_weight=min_edge_weight,
                measure_type=measure_type,
                min_community_size=min_community_size,
                n_threads=n_threads,
                text_embed_col=text_embed_col,
                image_embed_col=image_embed_col,
                message_id_col=message_id_col,
                username_col=username_col,
                show_progress_bar=show_progress_bar,
            )
            results[NETWORK_TICNET] = ticnet_path

        if network_type in (NETWORK_LICNET, NETWORK_BOTH):
            licnet_path = os.path.abspath(os.path.join(output_dir, "LiCNet.graphml"))
            if show_progress_bar:
                print(f"  → Computing LiCNet → {licnet_path}")
            compute_licnet(
                db_path=db_path,
                output_path=licnet_path,
                time_window=LICNET_TIME_WINDOW,
                text_threshold=text_threshold,
                img_threshold=img_threshold,
                min_edge_weight=LICNET_MIN_EDGE_WEIGHT,
                measure_type=measure_type,
                n_threads=n_threads,
                show_progress_bar=show_progress_bar,
            )
            results[NETWORK_LICNET] = licnet_path

    finally:
        # Clean up temporary database if we created one
        if _tmp_db and os.path.exists(_tmp_db):
            os.remove(_tmp_db)

    if show_progress_bar:
        print("Step 4/4: Done.")
        for name, path in results.items():
            print(f"  {name}: {path}")

    return results
