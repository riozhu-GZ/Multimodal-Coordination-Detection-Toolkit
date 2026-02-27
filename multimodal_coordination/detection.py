"""
Coordination detection algorithms for the Multimodal Coordination Detection Toolkit.

Implements the core time-window pairwise similarity search that identifies
accounts co-posting similar multimodal content, using parallelised SQLite queries.
"""

import math
import multiprocessing as mp
import sqlite3 as lite
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Callable, Iterable, List, Optional

import networkx as nx
import numpy as np

from .config import (
    DEFAULT_ASY_MIN_TIME_WINDOW,
    DEFAULT_IMG_THRESHOLD,
    DEFAULT_MIN_EDGE_WEIGHT,
    DEFAULT_N_THREADS,
    DEFAULT_TEXT_THRESHOLD,
    DEFAULT_TIME_WINDOW,
    SIM_MULTIMODAL_DISJOINT,
    VALID_SIM_TYPES,
)
from .database import COMMAND_MULTIMODAL_TABLE, COMMAND_TABLE


# ---------------------------------------------------------------------------
# Parallelised SQLite query execution
# ---------------------------------------------------------------------------

def _run_query(
    db_path, target_table, query, query_parameters, user_ids, sqlite_functions, lock
):
    """Run the coordination query on the subset of user_ids provided."""
    db = lite.connect(db_path)
    db.execute(
        f"""
        create temporary table local_network as
            select * from {target_table} limit 0;
        """
    )
    db.execute(
        """
        create temporary table user_id (
            user_id primary key
        );
        """
    )

    for func_name, (func, n_args) in sqlite_functions.items():
        db.create_function(func_name, n_args, func)

    with db:
        db.executemany(
            "insert into user_id values(?)", ((uid,) for uid in user_ids)
        )
        db.execute(f"insert into local_network {query}", query_parameters)

    with lock, db:
        db.execute(f"insert into {target_table} select * from local_network")


def parallelise_query_by_user_id(
    db_path: str,
    target_table: str,
    query: str,
    query_parameters: list,
    n_processes: int = DEFAULT_N_THREADS,
    sqlite_functions: dict = None,
    user_selection_query: str = "select distinct user_id from edge",
    user_selection_query_parameters: list = None,
    show_progress_bar: bool = False,
) -> int:
    """
    Execute a SQL query in parallel across user ID batches.

    Splits the user ID space into batches and runs each batch in a separate
    subprocess, writing results back to the shared target table under a lock.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    target_table : str
        Name of the table to write results into.
    query : str
        SQL SELECT query to run per batch (must reference ``user_id`` temp table).
    query_parameters : list
        Positional parameters for the query.
    n_processes : int
        Number of parallel worker processes.
    sqlite_functions : dict, optional
        Custom SQLite functions to register: {name: (callable, n_args)}.
    user_selection_query : str
        Query to retrieve the list of user IDs to partition over.
    user_selection_query_parameters : list, optional
        Parameters for the user selection query.
    show_progress_bar : bool
        Print progress to stdout.

    Returns
    -------
    int
        Number of completed batches.
    """
    manager = mp.Manager()
    lock = manager.Lock()
    pool = ProcessPoolExecutor(max_workers=n_processes)
    db = lite.connect(db_path)

    user_ids = [
        row[0]
        for row in db.execute(
            user_selection_query, user_selection_query_parameters or []
        )
    ]

    target_batches = n_processes * 10
    batch_size = max(math.floor(len(user_ids) / target_batches), 1)
    batches = [user_ids[i : i + batch_size] for i in range(0, len(user_ids), batch_size)]

    waiting = set()
    for batch in batches:
        waiting.add(
            pool.submit(
                _run_query,
                db_path,
                target_table,
                query,
                query_parameters,
                batch,
                sqlite_functions or {},
                lock,
            )
        )

    completed = 0
    while waiting:
        done, waiting = wait(waiting, return_when=FIRST_COMPLETED)
        for d in done:
            d.result()
            completed += 1
            if show_progress_bar and not (completed % max(1, math.ceil(len(batches) / 10))):
                print(f"Completed {completed} / {len(batches)} batches")

    if show_progress_bar:
        print(f"Completed {completed} / {len(batches)} batches")

    db.close()
    return completed


# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------

def cos_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity between two numpy vectors. Returns 0.0 for zero/None vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def multimodal_similarity(
    textemb_1: Optional[str],
    textemb_2: Optional[str],
    imgemb_1: Optional[str],
    imgemb_2: Optional[str],
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    img_threshold: float = DEFAULT_IMG_THRESHOLD,
    sim_measure_type: str = SIM_MULTIMODAL_DISJOINT,
) -> int:
    """
    Determine whether two posts are similar under the given measurement strategy.

    This function is registered as a SQLite user-defined function and called
    once per post-pair during the pairwise search.

    Parameters
    ----------
    textemb_1, textemb_2 : str or None
        Comma-separated float strings representing text embeddings.
    imgemb_1, imgemb_2 : str or None
        Comma-separated float strings representing image embeddings.
    text_threshold : float
        Minimum cosine similarity for text to be considered similar.
    img_threshold : float
        Minimum cosine similarity for images to be considered similar.
    sim_measure_type : str
        One of:
        - ``'multimodal_disjoint'``: similar if text OR image exceeds threshold
        - ``'text_only'``: only text similarity is considered
        - ``'image_only'``: only image similarity is considered
        - ``'multimodal_joint'``: concatenated embedding similarity

    Returns
    -------
    int
        1 if the similarity criterion is met, 0 otherwise.
    """
    if sim_measure_type not in VALID_SIM_TYPES:
        raise ValueError(
            f"Invalid sim_measure_type: '{sim_measure_type}'. "
            f"Must be one of {VALID_SIM_TYPES}."
        )

    def parse(vec: Optional[str]) -> Optional[np.ndarray]:
        if vec is None or not str(vec).strip():
            return None
        return np.fromstring(str(vec).strip("[]"), sep=",")

    t1, t2 = parse(textemb_1), parse(textemb_2)
    i1, i2 = parse(imgemb_1), parse(imgemb_2)

    if sim_measure_type == SIM_MULTIMODAL_DISJOINT:
        text_sim = cos_similarity(t1, t2) if t1 is not None and t2 is not None else 0.0
        img_sim = cos_similarity(i1, i2) if i1 is not None and i2 is not None else 0.0
        return 1 if (text_sim >= text_threshold or img_sim >= img_threshold) else 0

    elif sim_measure_type == "text_only":
        if t1 is not None and t2 is not None:
            return 1 if cos_similarity(t1, t2) >= text_threshold else 0
        return 0

    elif sim_measure_type == "image_only":
        if i1 is not None and i2 is not None:
            return 1 if cos_similarity(i1, i2) >= img_threshold else 0
        return 0

    elif sim_measure_type == "multimodal_joint":
        if t1 is not None and i1 is not None and t2 is not None and i2 is not None:
            joint_threshold = max(text_threshold, img_threshold)
            emb1 = np.concatenate([t1, i1])
            emb2 = np.concatenate([t2, i2])
            return 1 if cos_similarity(emb1, emb2) >= joint_threshold else 0
        return 0

    return 0


# ---------------------------------------------------------------------------
# Main coordination detection function
# ---------------------------------------------------------------------------

def compute_co_similar_tweet_multimodal(
    db_path: str,
    time_window: int = DEFAULT_TIME_WINDOW,
    asy_min_time_window: int = DEFAULT_ASY_MIN_TIME_WINDOW,
    n_threads: int = DEFAULT_N_THREADS,
    text_similarity_threshold: float = DEFAULT_TEXT_THRESHOLD,
    img_similarity_threshold: float = DEFAULT_IMG_THRESHOLD,
    min_edge_weight: int = DEFAULT_MIN_EDGE_WEIGHT,
    similarity_function: Callable = multimodal_similarity,
    measure_type: str = SIM_MULTIMODAL_DISJOINT,
    embed_type: str = "text_image",
    show_progress_bar: bool = False,
    network_type: str = "co_similar_multimodal_network",
) -> int:
    """
    Detect pairwise coordination edges between accounts.

    Finds all pairs of accounts (user_1, user_2) where user_1 posted content
    that is similar to content posted by user_2 within ``time_window`` seconds.
    Results are stored in a new table in the database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database (must be initialised).
    time_window : int
        Maximum time gap in seconds between similar posts to count as coordinated.
    asy_min_time_window : int
        Minimum time gap in seconds (use 0 for no lower bound).
    n_threads : int
        Number of parallel worker processes.
    text_similarity_threshold : float
        Cosine similarity threshold for text (0–1).
    img_similarity_threshold : float
        Cosine similarity threshold for images (0–1).
    min_edge_weight : int
        Minimum number of co-similar post pairs needed to form an edge.
    similarity_function : callable
        Function with signature ``(t1, t2, i1, i2, txt_thresh, img_thresh, type) -> int``.
    measure_type : str
        Similarity measurement strategy (see :func:`multimodal_similarity`).
    embed_type : str
        Which embeddings to require: ``'text'``, ``'image'``, or ``'text_image'``.
    show_progress_bar : bool
        Print batch progress to stdout.
    network_type : str
        Name of the output table in the database.

    Returns
    -------
    int
        Number of completed query batches.
    """
    if show_progress_bar:
        print(f"Computing coordination network (embed_type={embed_type}, "
              f"measure={measure_type}, time_window={time_window}s)...")

    db = lite.connect(db_path, isolation_level=None)
    db.create_function("similarity", 7, similarity_function)

    # Build index and result table
    db.executescript(
        f"""
        drop index if exists user_time;
        create index if not exists non_repost_user_time on edge(user_id, timestamp)
            where repost_id is null;
        create index if not exists to_tokenize on edge(message_id)
            where repost_id is null and token_set is null;
        create index if not exists timestamp on edge(timestamp);
        drop table if exists {network_type};
        create table {network_type} (
            user_1,
            user_2,
            message_groups_1,
            message_groups_2,
            weight,
            primary key (user_1, user_2)
        ) without rowid;
        """
    )
    db.execute("begin")

    # Build query based on required embedding types
    embed_conditions = {
        "text": "e_1.text_embed is not null and e_2.text_embed is not null",
        "image": "e_1.image_embed is not null and e_2.image_embed is not null",
        "text_image": (
            "e_1.image_embed is not null and e_2.image_embed is not null "
            "and e_1.text_embed is not null and e_2.text_embed is not null"
        ),
    }
    if embed_type not in embed_conditions:
        raise ValueError(
            f"Invalid embed_type: '{embed_type}'. Must be one of {list(embed_conditions)}."
        )

    embed_filter = embed_conditions[embed_type]

    query = f"""
        select
            e_1.user_id as user_1,
            e_2.user_id as user_2,
            group_concat(e_1.message_id, ',') as message_groups_1,
            group_concat(e_2.message_id, ',') as message_groups_2,
            count(distinct e_1.message_id) as weight
        from edge e_1 indexed by non_repost_user_time
        inner join edge e_2
            on e_2.timestamp between e_1.timestamp + ?6 and e_1.timestamp + ?1
        where
            e_1.repost_id is null
            and e_2.repost_id is null
            and {embed_filter}
            and similarity(
                e_1.text_embed, e_2.text_embed,
                e_1.image_embed, e_2.image_embed,
                ?3, ?4, ?5
            ) == 1
            and user_1 in (select user_id from user_id)
            and e_1.message_id != e_2.message_id
        group by e_1.user_id, e_2.user_id
        having weight >= ?2
    """

    user_selection_query = """
        select user_id
        from edge
        where repost_id is null
        group by user_id
        having count(*) >= ?
    """

    return parallelise_query_by_user_id(
        db_path,
        network_type,
        query,
        [
            time_window,
            min_edge_weight,
            text_similarity_threshold,
            img_similarity_threshold,
            measure_type,
            asy_min_time_window,
        ],
        n_processes=n_threads,
        sqlite_functions={"similarity": (similarity_function, 7)},
        user_selection_query=user_selection_query,
        user_selection_query_parameters=[min_edge_weight],
        show_progress_bar=show_progress_bar,
    )


# ---------------------------------------------------------------------------
# Graph loading utilities
# ---------------------------------------------------------------------------

def get_node_rows(db_path: str):
    """Yield (user_id, username) rows for all users in the edge table."""
    db = lite.connect(db_path)
    db.execute("create index if not exists user_time on edge(user_id, timestamp)")
    for user_id, in db.execute("select distinct user_id from edge"):
        username = db.execute(
            "select max(username) from edge where user_id = ?", [user_id]
        ).fetchone()[0]
        yield [user_id, username]


def get_edge_rows(db_path: str, command: str, loops: bool = False):
    """
    Return an iterator over the rows of the coordination result table.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    command : str
        Network command key (e.g. ``'co_similar_multimodal'``).
    loops : bool
        If False (default), exclude self-loops (user_1 == user_2).
    """
    if command not in COMMAND_TABLE and command not in COMMAND_MULTIMODAL_TABLE:
        raise ValueError(f"Unknown command: '{command}'.")

    table = COMMAND_TABLE.get(command) or COMMAND_MULTIMODAL_TABLE.get(command)
    loop_filter = "" if loops else "where user_1 != user_2"
    query = f"select *, '{command}' from {table} {loop_filter}"
    return lite.connect(db_path).execute(query)


def load_networkx_graph(
    db_path: str,
    command: str,
    emb_type: str = None,
    sim_measure_type: str = None,
    loops: bool = False,
    index: int = None,
) -> nx.DiGraph:
    """
    Load a coordination network from the database into a NetworkX directed graph.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    command : str
        Network command key (e.g. ``'co_similar_multimodal'``).
    emb_type : str, optional
        Embedding type label to annotate edge/node attributes.
    sim_measure_type : str, optional
        Similarity type label to annotate edge/node attributes.
    loops : bool
        Include self-loops.
    index : int, optional
        Window index label (for temporal windowing).

    Returns
    -------
    nx.DiGraph
        Graph with:
        - Nodes: user_id, attributes: username, network_type, message_i
        - Edges: weight, edge_type, edge_network, edges_message
    """
    network_name = "_".join(
        part for part in [command, emb_type, sim_measure_type] if part is not None
    )
    g = nx.DiGraph()
    nodes_dict = {}

    def _add_messages(d, user, msgs):
        if user in d:
            d[user] = [*d[user], *msgs]
        else:
            d[user] = msgs if isinstance(msgs, list) else [msgs]

    for user_1, user_2, m1, m2, weight, edge_type in get_edge_rows(db_path, command, loops):
        m1_list = str(m1).split(",")
        m2_list = str(m2).split(",")
        _add_messages(nodes_dict, user_1, m1_list)
        _add_messages(nodes_dict, user_2, m2_list)

        edges_message = ",".join(
            f"{m1_list[i]}-{m2_list[i]}" for i in range(min(len(m1_list), len(m2_list)))
        )
        edge_attrs = dict(
            weight=weight,
            edge_type=edge_type,
            edge_network=network_name,
            edges_message=edges_message,
        )
        if index is not None:
            edge_attrs["window_index"] = index
        g.add_edge(user_1, user_2, **edge_attrs)

    for user_id, username in get_node_rows(db_path):
        if user_id in g.nodes:
            node_attrs = {"username": username, "network_type": network_name}
            if index is not None:
                node_attrs["window_index"] = index
            for i, msg in enumerate(nodes_dict.get(user_id, [])):
                node_attrs[f"message_{i}"] = msg
            g.add_node(user_id, **node_attrs)

    return g


def output_graphml(
    db_path: str,
    command: str,
    output_file: str,
    loops: bool = False,
) -> None:
    """
    Write the coordination network to a GraphML file.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    command : str
        Network command key.
    output_file : str
        Destination path for the GraphML file.
    loops : bool
        Include self-loops.
    """
    graph = load_networkx_graph(db_path, command, loops=loops)
    nx.write_graphml(graph, output_file)
