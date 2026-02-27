"""
Network construction and export for the Multimodal Coordination Detection Toolkit.

Implements TiCNet and LiCNet coordination network builders on top of
the pairwise co-similar edge detection in :mod:`detection`.

TiCNet (Tweet-image Coordination Network)
    A strict coordination network that applies Leiden community detection
    to filter user groups, then builds a content network (indexed by
    embedding clusters) and composes it back into a user-level network.

LiCNet (Loose image Coordination Network)
    A simpler coordination network: direct edges between users who share
    similar multimodal content above a weight threshold within a time window.
"""

import itertools
import os
from collections import Counter
from datetime import datetime
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
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
    SIM_MULTIMODAL_DISJOINT,
    DEFAULT_CONTENT_COMMUNITY_MIN_SIZE,
    DEFAULT_EMBED_CLUSTER_MIN_SIZE,
)
from .detection import (
    compute_co_similar_tweet_multimodal,
    load_networkx_graph,
)

try:
    import igraph as ig
    import leidenalg
    from sentence_transformers import util as st_util
    _COMMUNITY_DEPS = True
except ImportError:
    _COMMUNITY_DEPS = False


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

def apply_leiden_community_detection(
    graph: nx.Graph,
    resolution: float = 0.1,
) -> dict:
    """
    Apply the Leiden algorithm for community detection on a user graph.

    Parameters
    ----------
    graph : nx.Graph or nx.DiGraph
        Input coordination graph (nodes = users).
    resolution : float
        CPM resolution parameter. Higher values → smaller, denser communities.

    Returns
    -------
    dict
        Mapping of {user_id: community_id}.

    Raises
    ------
    ImportError
        If leidenalg or igraph are not installed.
    """
    if not _COMMUNITY_DEPS:
        raise ImportError(
            "Community detection requires: pip install leidenalg igraph python-igraph"
        )
    g_ig = ig.Graph.from_networkx(graph)
    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.CPMVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
    )
    return {g_ig.vs[i]["_nx_name"]: m for i, m in enumerate(partition.membership)}


def filter_graph_by_community(
    graph: nx.Graph,
    community_mapping: dict,
    min_size: int = DEFAULT_MIN_COMMUNITY_SIZE,
) -> nx.Graph:
    """
    Return a subgraph keeping only communities larger than ``min_size``.

    Parameters
    ----------
    graph : nx.Graph or nx.DiGraph
        Input graph.
    community_mapping : dict
        {node: community_id} mapping.
    min_size : int
        Communities with <= min_size nodes are discarded.

    Returns
    -------
    nx.DiGraph
        Filtered graph.
    """
    community_counts = Counter(community_mapping.values())
    keep = {comm for comm, cnt in community_counts.items() if cnt > min_size}

    filtered = nx.DiGraph()
    for node, attrs in graph.nodes(data=True):
        if community_mapping.get(node) in keep:
            filtered.add_node(node, **attrs)
    for u, v, attrs in graph.edges(data=True):
        if community_mapping.get(u) in keep and community_mapping.get(v) in keep:
            filtered.add_edge(u, v, **attrs)

    return filtered


# ---------------------------------------------------------------------------
# Content network
# ---------------------------------------------------------------------------

def find_embed_clusters(
    keys: list,
    corpus_embeddings: list,
    threshold: float = 0.9,
    min_cluster_size: int = DEFAULT_EMBED_CLUSTER_MIN_SIZE,
) -> Tuple[dict, list]:
    """
    Cluster a set of embeddings using cosine-similarity community detection.

    Uses sentence-transformers' ``util.community_detection`` under the hood.

    Parameters
    ----------
    keys : list
        Identifier for each embedding (e.g. a hash).
    corpus_embeddings : list of array-like
        Embedding vectors, one per key.
    threshold : float
        Minimum cosine similarity to link two embeddings in the same cluster.
    min_cluster_size : int
        Discard clusters smaller than this.

    Returns
    -------
    cluster_dict : dict
        {key: cluster_id}. Singletons get cluster_id = 0.
    large_cluster_keys : list
        Keys belonging to clusters with more than 5 members.
    """
    if not _COMMUNITY_DEPS:
        raise ImportError(
            "Embedding clustering requires: pip install sentence-transformers"
        )
    corpus = np.array(corpus_embeddings)
    clusters = st_util.community_detection(
        corpus,
        threshold=threshold,
        min_community_size=min_cluster_size,
        batch_size=1024,
        show_progress_bar=False,
    )

    labels = [0] * len(keys)
    for cluster_id, cluster in enumerate(clusters, start=1):
        for idx in cluster:
            labels[idx] = cluster_id

    cluster_dict = dict(zip(keys, labels))
    cluster_count = Counter(cluster_dict.values())
    large_clusters = {cid for cid, cnt in cluster_count.items() if cnt > 5}
    large_cluster_keys = [k for k, cid in cluster_dict.items() if cid in large_clusters]

    return cluster_dict, large_cluster_keys


def build_content_network(
    network: nx.Graph,
    tweet_df: pd.DataFrame,
    text_embed_col: str = "text_emb_from_multi",
    image_embed_col: str = "image_embed",
    message_id_col: str = "id",
    username_col: str = "username",
    user_community_min_size: int = DEFAULT_CONTENT_COMMUNITY_MIN_SIZE,
    embed_cluster_min_size: int = DEFAULT_EMBED_CLUSTER_MIN_SIZE,
) -> Tuple[nx.Graph, nx.Graph]:
    """
    Build a content-level network from the user coordination edges.

    Indexes content by the hash of concatenated (text + image) embeddings,
    clusters similar content together, then retains only large connected
    components.

    Parameters
    ----------
    network : nx.Graph
        User-level coordination graph (edges carry ``edges_message`` attribute).
    tweet_df : pd.DataFrame
        DataFrame with post data including embedding columns and message IDs.
    text_embed_col, image_embed_col : str
        Column names for text and image embeddings.
    message_id_col, username_col : str
        Column name mappings.
    user_community_min_size : int
        Minimum component size to retain.
    embed_cluster_min_size : int
        Minimum cluster size for embedding clustering.

    Returns
    -------
    g_content : nx.Graph
        Full content network (all components).
    g_content_big : nx.Graph
        Filtered content network (large components only).
    """
    edges_df = pd.DataFrame(
        network.edges(data=True), columns=["Source", "Target", "Attributes"]
    )
    edges_df = pd.concat(
        [edges_df, pd.json_normalize(edges_df["Attributes"])], axis=1
    ).drop(columns="Attributes")

    def _parse_pairs(pairs: str) -> list:
        return [list(map(int, p.split("-"))) for p in str(pairs).split(",")]

    edges_df["edges_message"] = edges_df["edges_message"].apply(_parse_pairs)
    edges_df = edges_df.explode("edges_message")
    edge_pairs = edges_df["edges_message"].tolist()

    tweet_ids = list(set(itertools.chain(*edge_pairs)))
    content_df = tweet_df[tweet_df[message_id_col].isin(tweet_ids)].drop_duplicates(
        subset=message_id_col
    ).copy()

    # Hash of combined embedding as a content identity key
    content_df["combine_embed"] = content_df.apply(
        lambda r: np.concatenate([
            np.array(r[text_embed_col]) if not isinstance(r[text_embed_col], np.ndarray) else r[text_embed_col],
            np.array(r[image_embed_col]) if not isinstance(r[image_embed_col], np.ndarray) else r[image_embed_col],
        ]),
        axis=1,
    )
    content_df["embed_key"] = content_df["combine_embed"].apply(lambda e: hash(tuple(e)))

    id_to_embed_key = content_df.set_index(message_id_col)["embed_key"].to_dict()
    author_dict = content_df.set_index(message_id_col)[username_col].to_dict()
    embed_keys = content_df["embed_key"].tolist()
    embeddings = content_df["combine_embed"].tolist()

    cluster_dict, _ = find_embed_clusters(
        embed_keys, embeddings, min_cluster_size=embed_cluster_min_size
    )

    # Build content graph (nodes = content hashes, edges = co-post pairs)
    embed_pairs = [
        (id_to_embed_key[pair[0]], id_to_embed_key[pair[1]])
        for pair in edge_pairs
        if pair[0] in id_to_embed_key and pair[1] in id_to_embed_key
    ]
    pair_counts = Counter(embed_pairs)

    g_content = nx.Graph()
    for (src, tgt), weight in pair_counts.items():
        g_content.add_edge(src, tgt, weight=weight)

    def _node_attrs(embed_key):
        linked_ids = [
            str(mid) for mid, key in id_to_embed_key.items() if key == embed_key
        ]
        usernames = [author_dict[int(mid)] for mid in linked_ids if int(mid) in author_dict]
        return {
            "message_id": ",".join(linked_ids),
            "usernames": ",".join(usernames),
            "user_count": len(usernames),
            "embed_cluster_idx": cluster_dict.get(embed_key, 0),
        }

    for ek in set(embed_keys):
        if ek in g_content.nodes:
            g_content.nodes[ek].update(_node_attrs(ek))

    # Keep only large connected components
    components = list(nx.connected_components(g_content))
    large_nodes = {
        node
        for comp in components
        if len(comp) > user_community_min_size
        for node in comp
    }

    g_content_big = nx.Graph()
    for u, v, data in g_content.edges(data=True):
        if u in large_nodes and v in large_nodes:
            g_content_big.add_edge(u, v, **data)
    for ek in g_content_big.nodes:
        g_content_big.nodes[ek].update(_node_attrs(ek))

    return g_content, g_content_big


# ---------------------------------------------------------------------------
# User network composition
# ---------------------------------------------------------------------------

def build_fully_connected_user_network(graph: nx.Graph) -> nx.Graph:
    """
    Build a fully-connected user network per content cluster.

    For each embedding cluster, connect all users that shared content in
    that cluster to form a clique.

    Parameters
    ----------
    graph : nx.Graph
        Content network (nodes have ``embed_cluster_idx`` and ``usernames``).

    Returns
    -------
    nx.Graph
        Combined fully-connected user graph.
    """
    nodes_df = pd.DataFrame(graph.nodes(data=True), columns=["Node", "Attributes"])
    nodes_df = pd.concat(
        [nodes_df, pd.json_normalize(nodes_df["Attributes"])], axis=1
    ).drop(columns="Attributes")

    combined = nx.Graph()
    for cluster in nodes_df["embed_cluster_idx"].unique():
        cluster_rows = nodes_df[nodes_df["embed_cluster_idx"] == cluster]["usernames"].tolist()
        users = [u for row in cluster_rows for u in str(row).split(",")]
        combined = nx.compose(combined, nx.complete_graph(users))

    return combined


def convert_id_to_username_network(
    original_network: nx.Graph,
    network_type: str = "undirect",
) -> nx.Graph:
    """
    Rebuild a network with usernames as node identifiers instead of user IDs.

    Parameters
    ----------
    original_network : nx.Graph or nx.DiGraph
        Network whose nodes have a ``username`` attribute.
    network_type : str
        ``'direct'`` for DiGraph, ``'undirect'`` for Graph.

    Returns
    -------
    nx.Graph or nx.DiGraph
    """
    username_network = nx.DiGraph() if network_type == "direct" else nx.Graph()

    for uid, data in original_network.nodes(data=True):
        username = data.get("username")
        if username:
            username_network.add_node(username, **data)

    for u, v, data in original_network.edges(data=True):
        uname_u = original_network.nodes[u].get("username")
        uname_v = original_network.nodes[v].get("username")
        if uname_u and uname_v:
            username_network.add_edge(uname_u, uname_v, **data)

    return username_network


def compose_directed_networks(
    network_a: nx.DiGraph,
    network_b: nx.Graph,
) -> nx.DiGraph:
    """
    Compose two networks: keep nodes from B, edges from A where both endpoints exist in B.

    Parameters
    ----------
    network_a : nx.DiGraph
        Source of directed edges.
    network_b : nx.Graph
        Source of nodes (defines the node set to keep).

    Returns
    -------
    nx.DiGraph
    """
    composed = nx.DiGraph()
    composed.add_nodes_from(network_b.nodes(data=True))

    for u, v, data in network_a.edges(data=True):
        if u in network_b and v in network_b:
            composed.add_edge(u, v, **data)

    for node in composed.nodes:
        if node in network_a.nodes:
            composed.nodes[node].update(network_a.nodes[node])

    return composed


# ---------------------------------------------------------------------------
# TiCNet
# ---------------------------------------------------------------------------

def compute_ticnet(
    db_path: str,
    tweet_df: pd.DataFrame,
    output_path: str,
    time_window: int = DEFAULT_TIME_WINDOW,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    img_threshold: float = DEFAULT_IMG_THRESHOLD,
    min_edge_weight: int = DEFAULT_MIN_EDGE_WEIGHT,
    measure_type: str = SIM_MULTIMODAL_DISJOINT,
    min_community_size: int = DEFAULT_MIN_COMMUNITY_SIZE,
    n_threads: int = DEFAULT_N_THREADS,
    text_embed_col: str = "text_emb_from_multi",
    image_embed_col: str = "image_embed",
    message_id_col: str = "id",
    username_col: str = "username",
    show_progress_bar: bool = True,
) -> nx.Graph:
    """
    Compute the Tweet-image Coordination Network (TiCNet) and save as GraphML.

    Pipeline:
    1. Detect co-similar post pairs (time-windowed pairwise similarity).
    2. Apply Leiden community detection; filter small communities.
    3. Build content network (embedding-hash nodes).
    4. Reconstruct a user network from content clusters (fully-connected cliques).
    5. Compose the directed user edges from step 2 onto the user set from step 4.
    6. Write GraphML output.

    Parameters
    ----------
    db_path : str
        Path to the initialised SQLite database.
    tweet_df : pd.DataFrame
        Post DataFrame with embedding columns.
    output_path : str
        Destination path for the TiCNet GraphML file.
    time_window : int
        Time window in seconds for coordination detection.
    text_threshold, img_threshold : float
        Similarity thresholds (0–1).
    min_edge_weight : int
        Minimum co-similar pairs to form an edge.
    measure_type : str
        Similarity measurement strategy.
    min_community_size : int
        Minimum Leiden community size to retain.
    n_threads : int
        Parallel workers for detection.
    text_embed_col, image_embed_col : str
        Embedding column names in ``tweet_df``.
    message_id_col, username_col : str
        Column name mappings in ``tweet_df``.
    show_progress_bar : bool
        Print progress.

    Returns
    -------
    nx.Graph
        The computed TiCNet graph.
    """
    start = datetime.now()
    if show_progress_bar:
        print("=" * 30)
        print("Computing TiCNet...")

    compute_co_similar_tweet_multimodal(
        db_path,
        time_window=time_window,
        text_similarity_threshold=text_threshold,
        img_similarity_threshold=img_threshold,
        min_edge_weight=min_edge_weight,
        measure_type=measure_type,
        embed_type="text_image",
        n_threads=n_threads,
        show_progress_bar=show_progress_bar,
    )

    g_co = load_networkx_graph(
        db_path, "co_similar_multimodal",
        sim_measure_type=measure_type, emb_type="text_image",
    )
    if show_progress_bar:
        print(f"Co-similar network: {g_co.number_of_nodes()} nodes, {g_co.number_of_edges()} edges")

    community_mapping = apply_leiden_community_detection(g_co)
    nx.set_node_attributes(g_co, community_mapping, "community")

    filtered = filter_graph_by_community(g_co, community_mapping, min_size=min_community_size)
    if show_progress_bar:
        print(f"After community filtering: {filtered.number_of_nodes()} nodes, {filtered.number_of_edges()} edges")

    _, g_content_big = build_content_network(
        filtered, tweet_df,
        text_embed_col=text_embed_col,
        image_embed_col=image_embed_col,
        message_id_col=message_id_col,
        username_col=username_col,
    )
    if show_progress_bar:
        print(f"Content network: {g_content_big.number_of_nodes()} nodes, {g_content_big.number_of_edges()} edges")

    user_clique_net = build_fully_connected_user_network(g_content_big)
    user_directed = convert_id_to_username_network(filtered, "undirect")
    ticnet = compose_directed_networks(user_directed, user_clique_net)

    if show_progress_bar:
        print(f"TiCNet: {ticnet.number_of_nodes()} nodes, {ticnet.number_of_edges()} edges")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    nx.write_graphml(ticnet, output_path)

    elapsed = (datetime.now() - start).total_seconds() / 60
    if show_progress_bar:
        print(f"TiCNet saved to: {output_path}  ({elapsed:.2f} min)")
        print("=" * 30)

    return ticnet


# ---------------------------------------------------------------------------
# LiCNet
# ---------------------------------------------------------------------------

def compute_licnet(
    db_path: str,
    output_path: str,
    time_window: int = LICNET_TIME_WINDOW,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    img_threshold: float = DEFAULT_IMG_THRESHOLD,
    min_edge_weight: int = LICNET_MIN_EDGE_WEIGHT,
    measure_type: str = SIM_MULTIMODAL_DISJOINT,
    n_threads: int = DEFAULT_N_THREADS,
    show_progress_bar: bool = True,
) -> nx.DiGraph:
    """
    Compute the Loose image Coordination Network (LiCNet) and save as GraphML.

    Pipeline:
    1. Detect co-similar post pairs with a larger time window and higher
       minimum edge weight than TiCNet (looser but more evidence required).
    2. Load the resulting directed user graph.
    3. Write GraphML output.

    Parameters
    ----------
    db_path : str
        Path to the initialised SQLite database.
    output_path : str
        Destination path for the LiCNet GraphML file.
    time_window : int
        Time window in seconds (default: 1 hour).
    text_threshold, img_threshold : float
        Similarity thresholds (0–1).
    min_edge_weight : int
        Minimum co-similar pairs to form an edge (default: 5).
    measure_type : str
        Similarity measurement strategy.
    n_threads : int
        Parallel workers.
    show_progress_bar : bool
        Print progress.

    Returns
    -------
    nx.DiGraph
        The computed LiCNet graph.
    """
    start = datetime.now()
    if show_progress_bar:
        print("=" * 30)
        print("Computing LiCNet...")

    compute_co_similar_tweet_multimodal(
        db_path,
        time_window=time_window,
        asy_min_time_window=0,
        text_similarity_threshold=text_threshold,
        img_similarity_threshold=img_threshold,
        min_edge_weight=min_edge_weight,
        measure_type=measure_type,
        embed_type="text_image",
        n_threads=n_threads,
        show_progress_bar=show_progress_bar,
    )

    licnet = load_networkx_graph(
        db_path, "co_similar_multimodal",
        sim_measure_type=measure_type, emb_type="text_image",
    )

    if show_progress_bar:
        print(f"LiCNet: {licnet.number_of_nodes()} nodes, {licnet.number_of_edges()} edges")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    nx.write_graphml(licnet, output_path)

    elapsed = (datetime.now() - start).total_seconds() / 60
    if show_progress_bar:
        print(f"LiCNet saved to: {output_path}  ({elapsed:.2f} min)")
        print("=" * 30)

    return licnet
