# Multimodal Coordination Detection Toolkit

A Python toolkit for detecting **coordinated inauthentic behaviour** on social media by identifying accounts that co-post similar multimodal content (text + images) within time windows.

The toolkit produces coordination networks as **GraphML files**, ready for analysis in Gephi, Cytoscape, or NetworkX.

---

## Overview

Coordinated inauthentic behaviour (CIB) refers to groups of social media accounts acting in concert to amplify narratives, often by posting similar or identical content. This toolkit detects such coordination by:

1. Encoding post text and images into a shared embedding space using **CLIP** (`clip-ViT-B-32`).
2. Detecting account pairs that repeatedly share similar multimodal content within a configurable **time window**.
3. Building coordination networks where nodes are accounts and edges indicate detected coordination.

Two network types are supported:

| Network | Full name | Strictness | Best for |
|---|---|---|---|
| **TiCNet** | Tweet-image Coordination Network | Strict — multi-stage filtering | Exposing tightly organised, high-confidence coordination groups |
| **LiCNet** | Loose image Coordination Network | Lenient — single-stage edge weight | Broad exploratory detection across large, noisy datasets |

---

## Network Types

### TiCNet — Tweet-image Coordination Network

**Rationale.** Genuine coordinated campaigns often involve accounts that repeatedly share near-identical multimodal content (same caption + same image, or slight variations) within tight time windows. A single co-similar post pair can occur by chance; TiCNet is designed to surface only the most credible coordination by requiring evidence across *multiple layers* simultaneously — temporal proximity, semantic similarity, community membership, and content-cluster co-membership.

**Design goals:**
- High precision: minimise false positives by requiring convergent evidence
- Community structure: detected nodes should belong to dense, internally-connected groups
- Content traceability: each edge can be traced back to the specific post pairs that caused it

**Pipeline:**

```
1. Co-similar detection
   For every pair of accounts (u1, u2): count posts within time_window that
   exceed text_threshold OR img_threshold. Edge weight = number of such pairs.

2. Leiden community detection (CPM variant, weighted)
   Partition the co-similar user graph into communities. The CPM objective
   finds groups that are more densely connected internally than expected.

3. Community filtering
   Discard any community smaller than min_community_size. This removes
   isolated account pairs and small noise clusters.

4. Content network construction
   For each post-pair edge (p1, p2) in the filtered graph:
     - Represent p1 and p2 by the hash of their combined (text + image) embedding
     - Build a content graph where nodes are unique content identities and
       edges are co-post links
     - Cluster content nodes by embedding similarity (cosine ≥ 0.9)
     - Retain only large connected components (removes singleton content)

5. User-clique reconstruction
   For each content cluster, connect all accounts that contributed content
   in that cluster as a fully-connected clique. This captures the full
   group of accounts amplifying the same multimodal content.

6. Composition
   Intersect the directed co-similar edges (step 1) with the user set from
   step 5. Only accounts present in both layers appear in the final graph.
   The result preserves directionality (who copied whom first) while
   restricting to accounts confirmed by content clustering.
```

**Output graph:**
- Nodes = accounts; edges = directed coordination links
- Node attributes include `community` (Leiden group ID) and linked post IDs
- Edge `weight` = number of co-similar post pairs; `edges_message` = traceable post-pair list

**Key parameters:** `time_window`, `text_threshold`, `img_threshold`, `min_edge_weight`, `min_community_size`

**When to use:** When you need high-confidence results for downstream analysis, reporting, or publication. Expect a smaller, denser graph with fewer false positives.

---

### LiCNet — Loose image Coordination Network

**Rationale.** Not all coordinated behaviour is tight or organised. Loosely affiliated actors may share similar images or text across a broader time span without belonging to a single coherent campaign. LiCNet applies a single filtering criterion — edge weight — over a wider time window, trading precision for recall.

**Design goals:**
- High recall: surface any pair of accounts with repeated multimodal similarity, even loosely timed
- Simplicity: single-pass, no community detection, directly interpretable
- Scalability: faster than TiCNet; suitable as a first-pass filter before deeper analysis

**Pipeline:**

```
1. Co-similar detection (same as TiCNet step 1, but wider window)
   Default: time_window = 3600 s (1 hour), min_edge_weight = 5
   An edge (u1, u2) exists if u1 and u2 each have ≥ 5 post pairs that
   exceed the similarity threshold within any 1-hour window.

2. Load and export
   The co-similar graph is loaded directly as the output network.
   No community detection, content network, or clique reconstruction.
```

**Output graph:**
- Nodes = accounts; edges = directed coordination links
- Edge `weight` = number of co-similar post pairs
- Node attributes: `username`, linked post IDs

**Key parameters:** `time_window` (uses `LICNET_TIME_WINDOW = 3600`), `text_threshold`, `img_threshold`, `min_edge_weight` (uses `LICNET_MIN_EDGE_WEIGHT = 5`)

**When to use:** As a fast first-pass scan, when recall matters more than precision, or when you want to study the full landscape of potential coordination before applying stricter filters.

---

### Choosing between TiCNet and LiCNet

| Question | TiCNet | LiCNet |
|---|---|---|
| Do you need traceable evidence per edge? | Yes | No |
| Are you working with a noisy or organic dataset? | Better (stricter) | Noisier results |
| Do you want to identify specific campaign groups? | Yes — communities | No — pairwise only |
| Is speed a priority? | Slower (multi-stage) | Faster (single-stage) |
| Do you want to explore broadly first? | No | Yes |

Use `network_type="both"` to compute both in one run and compare.

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

---

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

### Minimal example CSV

```csv
id,author_id,username,timestamp,text_emb_from_multi,image_embed,retweet_id,reply_id,url
1001,42,alice,1700000000,"[0.12,-0.34,...]","[0.56,-0.78,...]",,,
1002,43,bob,1700000120,"[0.11,-0.33,...]","[0.55,-0.77,...]",,,https://example.com
1003,42,alice,1700001800,"[0.90,0.10,...]","[0.88,0.05,...]",,,
1004,44,carol,1700002000,,,"1001",,   ← repost, excluded from detection
```

Key points:
- Row 1001 and 1002 are by different accounts, within 120 s, with similar embeddings → **coordination edge detected**
- Row 1003 is by `alice` again → compared against all posts within the time window
- Row 1004 is a repost (`retweet_id` is set) → **skipped** in detection

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

## Algorithm Description

### Step 1 — Embedding (optional)

If raw images and text are provided, both are encoded using the **CLIP ViT-B/32** model via the `sentence-transformers` library. CLIP produces a shared embedding space for text and images, enabling cross-modal similarity comparisons. Text is truncated to 77 tokens (CLIP limit).

### Step 2 — Database ingestion

Posts (with embeddings) are written into a **SQLite database** using a chunked parallel ingestion pipeline. Reposts/retweets are tagged and excluded from coordination analysis (they are not independently authored content).

### Step 3 — Pairwise coordination detection

For each user, the toolkit finds all other users who posted similar content within the time window using a SQL self-join on the database. Similarity is computed via a **registered SQLite user-defined function** (Python callable):

```
For each pair (post_A by user_1, post_B by user_2):
    if |timestamp_B - timestamp_A| <= time_window:
        if similarity(text_A, text_B, image_A, image_B) >= threshold:
            weight(user_1, user_2) += 1
```

Similarity strategies (controlled by `measure_type`):

| Strategy | Description |
|---|---|
| `multimodal_disjoint` | Text OR image similarity exceeds threshold (default) |
| `text_only` | Only text similarity |
| `image_only` | Only image similarity |
| `multimodal_joint` | Cosine similarity of concatenated text+image embedding |

This step is parallelised across user ID batches via `multiprocessing`.

### Step 4a — TiCNet construction

See [TiCNet — Tweet-image Coordination Network](#ticnet--tweet-image-coordination-network) for a full description. In brief:

1. **Leiden community detection** on the co-similar user graph (CPM variant, weighted edges).
2. **Filter small communities** (keep communities with > `min_community_size` nodes).
3. **Content network**: index posts by hash of concatenated (text + image) embedding; cluster similar content; retain large connected components.
4. **User clique network**: for each content cluster, form a fully-connected subgraph of all contributing accounts.
5. **Compose**: intersect the directed co-similar edges (step 3) with the user set from step 4.

### Step 4b — LiCNet construction

See [LiCNet — Loose image Coordination Network](#licnet--loose-image-coordination-network) for a full description. In brief:

Loads the co-similar user graph (step 3 output) directly using a wider time window (`LICNET_TIME_WINDOW = 3600 s`) and higher minimum edge weight (`LICNET_MIN_EDGE_WEIGHT = 5`). No community detection or content network step — the edge-filtered graph is the output.

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
| `time_window` | `--time-window` | `3600` | Max time gap in seconds (TiCNet) |
| `text_threshold` | `--text-threshold` | `0.9` | Text cosine similarity threshold |
| `img_threshold` | `--img-threshold` | `0.8` | Image cosine similarity threshold |
| `min_edge_weight` | `--min-edge-weight` | `1` | Min co-similar pairs per edge |
| `min_community_size` | `--min-community-size` | `3` | Min Leiden community size (TiCNet) |
| `measure_type` | `--measure-type` | `"multimodal_disjoint"` | Similarity strategy |
| `n_threads` | `--n-threads` | `4` | Parallel worker processes |
| `db_path` | `--db-path` | `None` | SQLite path (temp if None) |

---

## Advanced: Raw Image Input

If your dataset contains only raw images and text (no precomputed embeddings), set `precomputed_embeddings=False` and provide the image directory:

```python
results = detect_coordination(
    posts_df=df,
    output_dir="./results",
    precomputed_embeddings=False,    # will generate embeddings
    images_dir="./images",           # images/<filename> will be resolved
    text_col="text",
    image_col="image_filename",      # relative filenames in this column
    network_type="TiCNet",
)
```

Or via CLI:

```bash
python -m multimodal_coordination \
    --input posts.csv \
    --output-dir ./results \
    --images-dir ./images \
    --text-col text \
    --image-col image_filename \
    --network-type TiCNet
```

CLIP embedding takes time for large datasets. Consider running this step once and caching the embeddings to avoid repeating it.

---

## Package Structure

```
multimodal_coordination/
    __init__.py       Public API — exports detect_coordination()
    config.py         Default constants (thresholds, model name, window sizes)
    embeddings.py     CLIP text + image embedding generation
    database.py       SQLite schema, ingestion pipeline, utilities
    detection.py      Pairwise similarity detection + graph loading
    networks.py       TiCNet and LiCNet network construction
    pipeline.py       End-to-end detect_coordination() function
    __main__.py       CLI entry point (python -m multimodal_coordination)
dataset/
    toy_dataset_for_test.csv   3000-post sample with precomputed CLIP embeddings
run_example.py        Minimal runnable demo (uv run python run_example.py)
pyproject.toml        Project metadata + dependencies (uv / pip)
requirements.txt      Flat requirements list (alternative to pyproject.toml)
```

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

## License

See [LICENSE](LICENSE).
