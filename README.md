# Multimodal Coordination Detection Toolkit

A Python toolkit for detecting **multimodal coordinated online behaviour** on social media by identifying accounts that co-post similar images or multimodal content (text + images) within time windows. More details about multimodal coordinated online behaviour can be found in https://doi.org/10.5204/thesis.eprints.263305.

The toolkit produces coordination networks as **GraphML files**, ready for analysis in Gephi.

---

## Overview

Multimodal Coordinated Online Behaviour refers to a group of accounts who co-appear to share highly similar multimodal content in pursuit of an intent. This toolkit detects such coordination by:

1. Encoding post text and images into a shared embedding space using **CLIP** (`clip-ViT-B-32`).
2. Detecting account pairs that repeatedly share similar multimodal content within a configurable **time window**.
3. Building coordination networks where nodes are accounts and edges indicate detected coordination.

The toolkit extends the Coordination Network Toolkit (CNT) (Graham et al., 2024) by integrating multimodal embeddings to enhance the detection of coordinated behaviours involving multimodal content. While CNT primarily focuses on identifying coordinated inauthentic behaviour (CIB) through textual and network-based signals, the extended toolkit incorporates modality-aware representations to capture patterns emerging from images and other non-textual content.

Building on empirically observed patterns of multimodal coordinated online behaviour, the toolkit introduces two distinct network typologies designed to differentiate between synchronous and asynchronous coordination dynamics. These network structures enable a more fine-grained analysis of how actors coordinate across time and modalities.

Two network types are supported:

| Network | Full name | Use for |
|---|---|---|
| **TiCNet** | Transverse-Intensive Coordination Network | Exposing tightly synchronised, organised, high-confidence coordination groups |
| **LiCNet** | Longitudinal-Intensive Coordination Network | Exploratory detection on CIB and Asynchronised Coordination |

---

## Installation

### Recommended: uv (fast, reproducible)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync

# Run any script inside the managed environment
uv run python run_example.py
uv run python -m multimodal_coordination --help
```

### Alternative: pip

```bash
pip install -r requirements.txt
python run_example.py
```

**Hardware note:** CLIP embedding generation runs on CPU by default. A GPU (CUDA) or Apple Silicon (MPS) accelerator is detected and used automatically if available. If you already have precomputed embeddings, no GPU is needed.

---

## Quick Start

### Python API

```python
import pandas as pd
from multimodal_coordination import detect_coordination

# Load posts (must include precomputed embeddings OR raw text + image paths)
df = pd.read_csv("dataset/toy_dataset_for_test.csv")

results = detect_coordination(
    posts_df=df,
    output_dir="./results",
    precomputed_embeddings=True,         # embeddings already in the CSV
    text_embed_col="text_emb_from_multi",
    image_embed_col="image_embed",
    network_type="both",                 # produce both TiCNet and LiCNet
)

print(results)
# {'TiCNet': '/abs/path/results/TiCNet.graphml',
#  'LiCNet': '/abs/path/results/LiCNet.graphml'}
```

Run the bundled example:

```bash
uv run python run_example.py
```

### Command-line Interface

```bash
# Quickest path — precomputed embeddings already in CSV
uv run python -m multimodal_coordination \
    --input dataset/toy_dataset_for_test.csv \
    --output-dir ./results \
    --precomputed-embeddings \
    --network-type both

# Full pipeline — generate embeddings from raw images + text
uv run python -m multimodal_coordination \
    --input posts.csv \
    --output-dir ./results \
    --images-dir ./images \
    --text-col text \
    --image-col image_filename \
    --network-type TiCNet

# Get all options
uv run python -m multimodal_coordination --help
```

---

## Input Data Format

The toolkit accepts a **CSV file** or a **pandas DataFrame**. Column names are fully configurable — the defaults match the bundled toy dataset, but you can remap any column via the API parameters or CLI flags.

### Overview

```
Path A — precomputed embeddings (no GPU needed):
    CSV  →  [parse embeddings]  →  detect_coordination(precomputed_embeddings=True)

Path B — raw images + text (requires CLIP / GPU recommended):
    CSV + image files  →  detect_coordination(precomputed_embeddings=False, images_dir=...)
```

### Column reference

#### Always required

| Role | Default column name | Type | Constraints | Description |
|---|---|---|---|---|
| Post ID | `id` | `int` or `str` | Must be **unique** per row | Identifies each post. Used to trace which posts form a coordination edge. |
| Account ID | `author_id` | `int` or `str` | Must be stable per account | Numeric or string account identifier. Nodes in the output network correspond to unique values of this column. |
| Username | `username` | `str` | — | Human-readable handle (e.g. `@alice`). Used as node label in the output graph. If an account has multiple usernames in the data, the lexicographically largest value is used. |
| Timestamp | `timestamp` | `float` or `int` | **Unix epoch seconds** | Post creation time. Controls time-window coordination detection. Must be consistent (not milliseconds, not a date string). |

#### Required when `precomputed_embeddings=True`

| Role | Default column name | Type | Shape | Description |
|---|---|---|---|---|
| Text embedding | `text_emb_from_multi` | see below | 512-d for CLIP ViT-B/32 | CLIP text embedding for the post text. Posts where this is `None`/`NaN` can still be matched via image similarity (`multimodal_disjoint` mode). |
| Image embedding | `image_embed` | see below | 512-d for CLIP ViT-B/32 | CLIP image embedding. Posts with no image should have `None`/`NaN` here. |

**Accepted embedding formats** (all automatically parsed):

| Format | Example |
|---|---|
| Python list of floats | `[0.12, -0.34, 0.56, ...]` |
| NumPy array | `np.array([0.12, -0.34, ...])` |
| Comma-separated string | `"0.12,-0.34,0.56"` |
| NumPy repr string (no commas) | `"[ 1.47e-01  1.09e-01 -2.60e-01 ...]"` ← toy dataset format |

> **Important:** Both embeddings must come from the **same model** (same architecture, same weights). The cosine similarity comparison is only meaningful when text and image embeddings share the same vector space. The toolkit is pre-configured for CLIP ViT-B/32 (512 dimensions), but any model producing fixed-length L2-normalized vectors will work.

#### Required when `precomputed_embeddings=False`

| Role | Default column name | Type | Description |
|---|---|---|---|
| Post text | `text` | `str` or `NaN` | Raw text of the post. `NaN` rows are assigned a `None` text embedding and can still be matched by image. Truncated to 77 tokens (CLIP limit). |
| Image filename | `image_path` | `str` or `NaN` | Filename relative to `images_dir`, or an absolute path. Supported formats: JPEG, PNG, GIF (first frame), BMP, TIFF, WebP. `NaN` rows get a `None` image embedding. |

#### Optional

| Role | Default column name | Type | Description |
|---|---|---|---|
| Repost/retweet ID | `retweet_id` | `int`, `str`, or `NaN` | Parent post ID if this post is a repost of another. **Reposts are excluded from coordination detection** — only original posts authored by the account are compared. Set to `NaN` for original posts. |
| Reply ID | `reply_id` | `int`, `str`, or `NaN` | Parent post ID if this post is a reply. Replies are **included** in coordination detection (unlike reposts). |
| URLs | `url` | `str` or `NaN` | Space-separated list of URLs in the post body. Stored in the database but not currently used for coordination detection. |

---

## Output Format

Each output file is a **GraphML** file representing the coordination network.

### Node attributes

| Attribute | Type | Description |
|---|---|---|
| `username` | str | Account handle |
| `network_type` | str | Network name (e.g. `co_similar_multimodal_text_image_multimodal_disjoint`) |
| `community` | int | Leiden community ID (TiCNet only) |
| `message_0`, `message_1`, ... | str | Post IDs associated with this account |

### Edge attributes

| Attribute | Type | Description |
|---|---|---|
| `weight` | int | Number of co-similar post pairs |
| `edge_type` | str | Coordination command (e.g. `co_similar_multimodal`) |
| `edge_network` | str | Full network identifier string |
| `edges_message` | str | Comma-separated `postA-postB` pairs that constitute the edge |

---

## Parameter Reference

All parameters of `detect_coordination()` and their CLI equivalents:

| Python parameter | CLI flag | Default | Description |
|---|---|---|---|
| `precomputed_embeddings` | `--precomputed-embeddings` | `False` | Skip CLIP generation |
| `images_dir` | `--images-dir` | `None` | Base dir for image files |
| `text_col` | `--text-col` | `"text"` | Raw text column |
| `image_col` | `--image-col` | `"image_path"` | Image path column |
| `text_embed_col` | `--text-embed-col` | `"text_emb_from_multi"` | Text embedding column |
| `image_embed_col` | `--image-embed-col` | `"image_embed"` | Image embedding column |
| `user_id_col` | `--user-id-col` | `"author_id"` | User ID column |
| `username_col` | `--username-col` | `"username"` | Username column |
| `message_id_col` | `--message-id-col` | `"id"` | Post ID column |
| `timestamp_col` | `--timestamp-col` | `"timestamp"` | Timestamp column |
| `repost_id_col` | `--repost-id-col` | `"retweet_id"` | Repost/retweet ID column |
| `reply_id_col` | `--reply-id-col` | `"reply_id"` | Reply ID column |
| `url_col` | `--url-col` | `"url"` | URL column |
| `network_type` | `--network-type` | `"both"` | `"TiCNet"`, `"LiCNet"`, or `"both"` |
| `time_window` | `--time-window` | `60` | Max time gap in seconds for TiCNet (LiCNet always uses 3600 s) |
| `text_threshold` | `--text-threshold` | `0.9` | Text cosine similarity threshold |
| `img_threshold` | `--img-threshold` | `0.8` | Image cosine similarity threshold |
| `min_edge_weight` | `--min-edge-weight` | `1` | Min co-similar pairs per edge |
| `min_community_size` | `--min-community-size` | `3` | Min Leiden community size (TiCNet) |
| `measure_type` | `--measure-type` | `"image_only"` | Similarity strategy |
| `n_threads` | `--n-threads` | `4` | Parallel worker processes |
| `db_path` | `--db-path` | `None` | SQLite path (temp if None) |

---

## Dependencies

| Package | Purpose |
|---|---|
| `networkx` | Graph construction and GraphML export |
| `pandas`, `numpy` | Data handling |
| `sentence-transformers` | CLIP model loading and encoding |
| `torch` | Tensor operations (GPU/MPS/CPU) |
| `Pillow` | Image loading |
| `leidenalg`, `igraph` | Leiden community detection (TiCNet) |
| `tqdm` | Progress bars |

---

> **Reference:** Graham, Timothy; QUT Digital Observatory; (2020): Coordination Network Toolkit. Queensland University of Technology. (Software) https://doi.org/10.25912/RDF_1632782596538

