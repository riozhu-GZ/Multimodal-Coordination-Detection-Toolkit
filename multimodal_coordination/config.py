"""
Default configuration constants for the Multimodal Coordination Detection Toolkit.
"""

# --- Embedding model ---
CLIP_MODEL_NAME = "clip-ViT-B-32"
CLIP_TOKEN_LIMIT = 77          # CLIP tokenizer hard limit
TEXT_BATCH_SIZE = 50
IMAGE_BATCH_SIZE = 128

# --- Time-window detection defaults ---
DEFAULT_TIME_WINDOW = 3600           # 1 hour in seconds (generic default)
DEFAULT_ASY_MIN_TIME_WINDOW = 0      # minimum asymmetric window (0 = no lower bound)
DEFAULT_TEXT_THRESHOLD = 0.9         # cosine similarity threshold for text
DEFAULT_IMG_THRESHOLD = 0.8          # cosine similarity threshold for images
DEFAULT_MIN_EDGE_WEIGHT = 1          # minimum coordination count to form an edge

# --- TiCNet specific ---
# TiCNet uses a strict 60-second window to detect synchronous CIB (accounts
# posting visually identical images within seconds of each other).
TICNET_TIME_WINDOW = 60              # 60 seconds — strict CIB detection
TICNET_MEASURE_TYPE = "image_only"   # TiCNet detects by image similarity only

# --- LiCNet specific ---
LICNET_TIME_WINDOW = 60 * 60         # 1 hour — covers both CIB and async coordination
LICNET_MIN_EDGE_WEIGHT = 5
LICNET_MEASURE_TYPE = "image_only"   # LiCNet also detects by image similarity only

# --- Community filtering defaults ---
DEFAULT_MIN_COMMUNITY_SIZE = 3       # filter communities smaller than this
DEFAULT_EMBED_CLUSTER_MIN_SIZE = 2   # minimum cluster size for content network
DEFAULT_CONTENT_COMMUNITY_MIN_SIZE = 5

# --- Multiprocessing ---
DEFAULT_N_THREADS = 4

# --- Similarity measure types ---
SIM_MULTIMODAL_DISJOINT = "multimodal_disjoint"   # text OR image similarity
SIM_TEXT_ONLY = "text_only"
SIM_IMAGE_ONLY = "image_only"
SIM_MULTIMODAL_JOINT = "multimodal_joint"          # concatenated text+image similarity

VALID_SIM_TYPES = [
    SIM_MULTIMODAL_DISJOINT,
    SIM_TEXT_ONLY,
    SIM_IMAGE_ONLY,
    SIM_MULTIMODAL_JOINT,
]

# --- Network types ---
NETWORK_TICNET = "TiCNet"
NETWORK_LICNET = "LiCNet"
NETWORK_BOTH = "both"

VALID_NETWORK_TYPES = [NETWORK_TICNET, NETWORK_LICNET, NETWORK_BOTH]

# --- Database schema ---
DB_VERSION = 1
