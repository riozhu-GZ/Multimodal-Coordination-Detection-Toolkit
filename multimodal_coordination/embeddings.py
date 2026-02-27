"""
Embedding generation for the Multimodal Coordination Detection Toolkit.

Uses CLIP (clip-ViT-B-32) via sentence-transformers to produce aligned
text and image embeddings in the same vector space, enabling cross-modal
similarity comparisons.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import (
    CLIP_MODEL_NAME,
    CLIP_TOKEN_LIMIT,
    TEXT_BATCH_SIZE,
    IMAGE_BATCH_SIZE,
)

try:
    import torch
    from sentence_transformers import SentenceTransformer
    from PIL import Image
    _DEPS_AVAILABLE = True
except ImportError as e:
    _DEPS_AVAILABLE = False
    _IMPORT_ERROR = e


def _get_device() -> str:
    """Return the best available compute device."""
    if not _DEPS_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip_model(model_name: str = CLIP_MODEL_NAME) -> "SentenceTransformer":
    """
    Load the CLIP SentenceTransformer model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: 'clip-ViT-B-32'.

    Returns
    -------
    SentenceTransformer
        Loaded model placed on the best available device.

    Raises
    ------
    ImportError
        If sentence-transformers, torch, or Pillow are not installed.
    """
    if not _DEPS_AVAILABLE:
        raise ImportError(
            f"Embedding dependencies are not installed: {_IMPORT_ERROR}\n"
            "Install them with: pip install sentence-transformers torch Pillow"
        ) from _IMPORT_ERROR

    device = _get_device()
    print(f"Loading CLIP model '{model_name}' on device: {device}")
    model = SentenceTransformer(model_name, device=device)
    return model


def get_image_embeddings(
    image_paths: List[str],
    model: "SentenceTransformer",
    batch_size: int = IMAGE_BATCH_SIZE,
) -> List[Optional[List[float]]]:
    """
    Generate CLIP image embeddings for a list of image file paths.

    Parameters
    ----------
    image_paths : list of str
        Paths to image files (jpg, png, jpeg supported).
    model : SentenceTransformer
        A loaded CLIP SentenceTransformer model.
    batch_size : int
        Number of images per encoding batch.

    Returns
    -------
    list
        List of embedding vectors (as Python lists of float) in the same
        order as `image_paths`. Entries for missing/unreadable files are None.
    """
    device = model.device if hasattr(model, "device") else _get_device()

    results: List[Optional[List[float]]] = []
    valid_indices = []
    valid_images = []

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            valid_images.append(img)
            valid_indices.append(i)
        except Exception:
            results.append(None)

    # Pre-fill results list with None placeholders for invalid entries
    results = [None] * len(image_paths)

    if valid_images:
        embeddings = model.encode(
            valid_images,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=str(device),
        )
        embedding_list = embeddings.cpu().numpy().tolist()
        for idx, emb in zip(valid_indices, embedding_list):
            results[idx] = emb

    return results


def get_text_embeddings(
    texts: List[Optional[str]],
    model: "SentenceTransformer",
    batch_size: int = TEXT_BATCH_SIZE,
) -> List[Optional[List[float]]]:
    """
    Generate CLIP text embeddings for a list of strings.

    Handles None/NaN entries and automatically truncates to the CLIP
    token limit (77 tokens).

    Parameters
    ----------
    texts : list of str or None
        Input texts. None/NaN values yield a None embedding.
    model : SentenceTransformer
        A loaded CLIP SentenceTransformer model.
    batch_size : int
        Number of texts per encoding batch.

    Returns
    -------
    list
        List of embedding vectors (as Python lists of float).
        None for empty or NaN entries.
    """
    device = model.device if hasattr(model, "device") else _get_device()
    tokenizer = model._first_module().processor.tokenizer

    def _truncate(text: str) -> str:
        tokens = tokenizer.encode(text)
        if len(tokens) > CLIP_TOKEN_LIMIT:
            truncated = tokens[1 : CLIP_TOKEN_LIMIT - 1]
            return _truncate(tokenizer.decode(truncated))
        return text

    results: List[Optional[List[float]]] = [None] * len(texts)
    valid_indices = []
    valid_texts = []

    for i, text in enumerate(texts):
        if pd.notna(text) and isinstance(text, str) and text.strip():
            valid_texts.append(_truncate(text.strip()))
            valid_indices.append(i)

    if valid_texts:
        # Process in batches
        for batch_start in range(0, len(valid_texts), batch_size):
            batch = valid_texts[batch_start : batch_start + batch_size]
            batch_indices = valid_indices[batch_start : batch_start + batch_size]
            embeddings = model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=str(device),
            )
            embedding_list = embeddings.cpu().numpy().tolist()
            for idx, emb in zip(batch_indices, embedding_list):
                results[idx] = emb

    return results


def embed_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    image_col: str = "image_path",
    images_dir: Optional[str] = None,
    model: Optional["SentenceTransformer"] = None,
) -> pd.DataFrame:
    """
    Add `text_embed` and `image_embed` columns to a DataFrame by running CLIP.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must contain `text_col` and `image_col` columns.
    text_col : str
        Column with post text.
    image_col : str
        Column with image filenames (relative to `images_dir`) or absolute paths.
    images_dir : str, optional
        Base directory for image files. If provided, `image_col` values are
        joined to this directory. If None, `image_col` must be absolute paths.
    model : SentenceTransformer, optional
        Pre-loaded CLIP model. If None, one is loaded automatically.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with added columns:
        - ``text_embed``: list of float (CLIP text embedding)
        - ``image_embed``: list of float (CLIP image embedding)
    """
    if model is None:
        model = load_clip_model()

    df = df.copy()

    # Resolve image paths
    if images_dir is not None:
        image_paths = [
            os.path.join(images_dir, str(p)) if pd.notna(p) else None
            for p in df[image_col]
        ]
    else:
        image_paths = [str(p) if pd.notna(p) else None for p in df[image_col]]

    print("Generating text embeddings...")
    df["text_embed"] = get_text_embeddings(df[text_col].tolist(), model)

    print("Generating image embeddings...")
    # Filter out None paths before calling get_image_embeddings
    valid_image_paths = [p for p in image_paths if p is not None]
    valid_indices = [i for i, p in enumerate(image_paths) if p is not None]

    image_embeds = [None] * len(image_paths)
    if valid_image_paths:
        valid_embeds = get_image_embeddings(valid_image_paths, model)
        for idx, emb in zip(valid_indices, valid_embeds):
            image_embeds[idx] = emb

    df["image_embed"] = image_embeds

    return df
