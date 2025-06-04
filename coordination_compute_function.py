import sqlite3 as lite
import multiprocessing as mp
from concurrent.futures import (
    ProcessPoolExecutor,
    wait,
    FIRST_COMPLETED,
)
import math
from typing import Callable, Iterable, List




def _run_query(
    db_path, target_table, query, query_parameters, user_ids, sqlite_functions, lock
):
    """Run the target query on the subset of user_ids provided."""

    db = lite.connect(db_path)
    db.execute(
        f"""
        create temporary table local_network as
            select *
            from {target_table}
            limit 0
        ;
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

    # Note that this part is writing to temporary databases, which are independent
    # between processes, so this part can always be done in parallel
    with db:
        db.executemany(
            "insert into user_id values(?)", ((user_id,) for user_id in user_ids)
        )

        db.execute(
            f"""
            insert into local_network
                {query}
            """,
            query_parameters,
        )

    # This part requires the lock because it's writing back to the shared database.
    # Alternatively this could just spin on busy, so we don't need to worry about
    # the lock...
    with lock, db:
        db.execute(f"insert into {target_table} select * from local_network")

def parallise_query_by_user_id(
    db_path,
    target_table,
    query,
    query_parameters,
    n_processes=4,
    sqlite_functions=None,
    user_selection_query="select distinct user_id from edge",
    user_selection_query_parameters=None,
    show_progress_bar=False
):

    manager = mp.Manager()
    lock = manager.Lock()
    pool = ProcessPoolExecutor(max_workers=n_processes)
    db = lite.connect(db_path)
    waiting = set()
    count, completed, submitted = 0, 0, 0
    user_ids = [
        row[0]
        for row in db.execute(
            user_selection_query, user_selection_query_parameters or []
        )
    ]
    target_batches = n_processes * 10
    batch_size = max(math.floor(len(user_ids) / target_batches), 1)

    # Generate batches of user_ids
    batches = [
        user_ids[i : i + batch_size] for i in range(0, len(user_ids), batch_size)
    ]

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

    while waiting:
        done, waiting = wait(waiting, return_when=FIRST_COMPLETED)
        for d in done:
            d.result()
            completed += 1

            if show_progress_bar:
                if not (completed % math.ceil(len(batches) / 10)):
                    print(f"Completed {completed} / {len(batches)}")
    if show_progress_bar:
        print(f"Completed {completed} / {len(batches)}")

    db.close()


    return completed





import numpy as np
import sys
import re


import numpy as np
from typing import Optional


def cos_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two numpy vectors."""
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
    text_threshold: float = 0.0,
    img_threshold: float = 0.0,
    sim_measure_type: str = 'multimodal_disjoint'
) -> int:
    """
    Determine multimodal similarity based on the specified measurement type.

    Parameters:
    - textemb_1, textemb_2, imgemb_1, imgemb_2: string representations of embeddings (comma-separated floats)
    - text_threshold: minimum threshold for text similarity
    - img_threshold: minimum threshold for image similarity
    - sim_measure_type: one of ['multimodal_disjoint', 'text_only', 'image_only', 'multimodal_joint']

    Returns:
    - switch: 1 if similarity criteria are met, 0 otherwise
    """
    valid_types = ['multimodal_disjoint', 'text_only', 'image_only', 'multimodal_joint']
    if sim_measure_type not in valid_types:
        raise ValueError(f"Invalid sim_measure_type: {sim_measure_type}. Must be one of {valid_types}")

    def parse_embedding(vec: Optional[str]) -> Optional[np.ndarray]:
        if vec is None or not vec.strip():
            return None
        return np.fromstring(vec.strip('[]'), sep=',')

    # Convert embeddings
    textemb_1 = parse_embedding(textemb_1)
    textemb_2 = parse_embedding(textemb_2)
    imgemb_1 = parse_embedding(imgemb_1)
    imgemb_2 = parse_embedding(imgemb_2)

    # Initialize switch
    switch = 0

    if sim_measure_type == 'multimodal_disjoint':
        text_sim = cos_similarity(textemb_1, textemb_2) if textemb_1 is not None and textemb_2 is not None else 0
        img_sim = cos_similarity(imgemb_1, imgemb_2) if imgemb_1 is not None and imgemb_2 is not None else 0

        if (text_sim and text_sim >= text_threshold) and (img_sim and img_sim >= img_threshold):
            switch = 1
        elif (text_sim and text_sim >= text_threshold) or (img_sim and img_sim >= img_threshold):
            switch = 1

    elif sim_measure_type == 'image_only':
        if imgemb_1 is not None and imgemb_2 is not None:
            img_sim = cos_similarity(imgemb_1, imgemb_2)
            if img_sim >= img_threshold:
                switch = 1

    elif sim_measure_type == 'text_only':
        if textemb_1 is not None and textemb_2 is not None:
            text_sim = cos_similarity(textemb_1, textemb_2)
            if text_sim >= text_threshold:
                switch = 1

    elif sim_measure_type == 'multimodal_joint':
        if textemb_1 is not None and imgemb_1 is not None and textemb_2 is not None and imgemb_2 is not None:
            emb_1 = np.concatenate([textemb_1, imgemb_1])
            emb_2 = np.concatenate([textemb_2, imgemb_2])
            joint_sim = cos_similarity(emb_1, emb_2)
            if joint_sim >= img_threshold:  # Assuming img_threshold used for joint similarity
                switch = 1

    return switch




def compute_co_similar_tweet_multimodal(
    db_path,
    time_window,
    asy_min_time_window = 0,
    n_threads=4,
    text_similarity_threshold = .9,
    img_similarity_threshold = .8,
    min_edge_weight=1,
    similarity_function: Callable = multimodal_similarity,
    measure_type = 'disjoint',
    embed_type = 'image',
    show_progress_bar=False,
    network_type = 'co_similar_multimodal_network'
):

    if show_progress_bar:
        print("computing co_similar_tweet_multimodal network")

    db = lite.connect(db_path, isolation_level=None)
    db.create_function("similarity", 7, similarity_function)

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
    query = None
    if embed_type == 'text':
          query = """
            select
                e_1.user_id as user_1,
                e_2.user_id as user_2,
                group_concat(e_1.message_id, ',') as 'message_groups_1',
                group_concat(e_2.message_id, ',') as 'message_groups_2',
                count(distinct e_1.message_id) as weight
            from edge e_1 indexed by non_repost_user_time
            inner join edge e_2
                -- Length filtering of the messages
                on e_2.timestamp between e_1.timestamp+?6 and e_1.timestamp + ?1
            -- Note that this will only work where the Python similarity function has been
            -- registered on the connection - this is not a SQLite native function.
            where
                e_1.repost_id is null
                and e_2.repost_id is null
                and e_1.text_embed is not null
                and e_2.text_embed is not null
                and similarity(e_1.text_embed, e_2.text_embed, e_1.image_embed, e_2.image_embed, ?3, ?4, ?5) == 1
                and user_1 in (select user_id from user_id)
                and e_1.message_id != e_2.message_id
            group by e_1.user_id, e_2.user_id
            having weight >= ?2
        """
    elif embed_type == 'image':
          query = """
            select
                e_1.user_id as user_1,
                e_2.user_id as user_2,
                group_concat(e_1.message_id, ',') as 'message_groups_1',
                group_concat(e_2.message_id, ',') as 'message_groups_2',
                count(distinct e_1.message_id) as weight
            from edge e_1 indexed by non_repost_user_time
            inner join edge e_2
                -- Length filtering of the messages
                on e_2.timestamp between e_1.timestamp+?6 and e_1.timestamp + ?1
            -- Note that this will only work where the Python similarity function has been
            -- registered on the connection - this is not a SQLite native function.
            where
                e_1.repost_id is null
                and e_2.repost_id is null
                and e_1.image_embed is not null
                and e_2.image_embed is not null
                and similarity(e_1.text_embed, e_2.text_embed, e_1.image_embed, e_2.image_embed, ?3, ?4, ?5) == 1
                and user_1 in (select user_id from user_id)
                and e_1.message_id != e_2.message_id
            group by e_1.user_id, e_2.user_id
            having weight >= ?2
        """
    elif embed_type == 'text_image':
          query = """
              select
                  e_1.user_id as user_1,
                  e_2.user_id as user_2,
                  group_concat(e_1.message_id, ',') as 'message_groups_1',
                  group_concat(e_2.message_id, ',') as 'message_groups_2',
                  count(distinct e_1.message_id) as weight
              from edge e_1 indexed by non_repost_user_time
              inner join edge e_2
                  -- Length filtering of the messages
                  on e_2.timestamp between e_1.timestamp+?6 and e_1.timestamp + ?1
              -- Note that this will only work where the Python similarity function has been
              -- registered on the connection - this is not a SQLite native function.
              where
                  e_1.repost_id is null
                  and e_2.repost_id is null
                  and e_1.image_embed is not null
                  and e_2.image_embed is not null
                  and e_1.text_embed is not null
                  and e_2.text_embed is not null
                  and similarity(e_1.text_embed, e_2.text_embed, e_1.image_embed, e_2.image_embed, ?3, ?4, ?5) == 1
                  and user_1 in (select user_id from user_id)
                  and e_1.message_id != e_2.message_id
              group by e_1.user_id, e_2.user_id
              having weight >= ?2
          """

    if query == None:
        print('No working')

    # Optimisation - a user can never have an edge if the account doesn't have more
    # then min_edge_weight non-repost messages in the dataset. This is the same as
    # co-tweet behaviour
    user_selection_query = """
        select
            user_id
        from edge
        where repost_id is null
        group by user_id
        having count(*) >= ?
    """

    return parallise_query_by_user_id(
        db_path,
        network_type,
        query,
        [time_window, min_edge_weight, text_similarity_threshold, img_similarity_threshold, measure_type, asy_min_time_window],
        n_processes=n_threads,
        sqlite_functions={"similarity": (similarity_function, 7)},
        user_selection_query=user_selection_query,
        user_selection_query_parameters=[min_edge_weight],
    )




def get_node_rows(db_path):
    """Get an iterator of rows in the node table."""
    db = lite.connect(db_path)

    db.execute("create index if not exists user_time on edge(user_id, timestamp)")

    user_ids = (row[0] for row in db.execute("select distinct user_id from edge"))

    for user_id in user_ids:

        username = db.execute(
            "select max(username) from edge where user_id = ?", [user_id]
        ).fetchone()[0]

        yield [user_id, username]


def get_edge_rows(db_path, command, loops=False):
    """
    Return an iterator over the rows of the source table in the given database file.

    """

    if command not in COMMAND_TABLE and command not in COMMAND_MULTIMODAL_TABLE:
        raise ValueError(f"No known command for table {command}.")

    table = None
    if command in COMMAND_TABLE:
      table = COMMAND_TABLE[command]
    elif command in COMMAND_MULTIMODAL_TABLE:
      table = COMMAND_MULTIMODAL_TABLE[command]

    if loops:
        query = f"select *, '{command}' from {table}"
    else:
        query = f"select *, '{command}' from {table} where user_1 != user_2"

    db = lite.connect(db_path)

    rows = db.execute(query)

    return rows


def output_graphml(db_path, command, output_file, loops=False):
    """
    Output a graphml file, representing the nodes and edges of the given table.

    """

    graph = load_networkx_graph(db_path, command, loops=loops)

    nx.write_graphml(graph, output_file)


def load_networkx_graph(db_path, command, emb_type=None, sim_measure_type=None, loops=False, index=None):
    """Return a networkx graph object representing the given source table."""


    network_name = '_'.join([i for i in [command, emb_type, sim_measure_type] if i is not None])
    g = nx.DiGraph()

    edges = get_edge_rows(db_path, command)

    nodes_dict = {}
    def add_user_info(user_dict, user, info):
        if user in user_dict:
            # User exists, append the info to the existing list
            user_dict[user] = [*user_dict[user], *info]
        else:
            # User does not exist, create a new list with the info
            if isinstance(info, list):
              user_dict[user] = info
            else:
              user_dict[user] = [info]

    # Add the edges
    for user_1, user_2, m_1, m_2, weight, edge_type in edges:

        m_1 = m_1.split(',')
        m_2 = m_2.split(',')

        add_user_info(nodes_dict, user_1, m_1)
        add_user_info(nodes_dict, user_2, m_2)

        edges_message = ','.join([str(m_1[idx])+'-'+str(m_2[idx]) for idx, i in enumerate(m_1)])
        if index is not None:
            g.add_edge(user_1, user_2, weight=weight, edge_type=edge_type, edge_network=network_name, edges_message=edges_message, window_index = index)
        else:
            g.add_edge(user_1, user_2, weight=weight, edge_type=edge_type, edge_network=network_name, edges_message=edges_message)


    # Add the node annotations
    nodes = get_node_rows(db_path)
    for row in nodes:
        user_id = row[0]

        # Only add the node annotations if the node is present from an edge
        if user_id in g.nodes:
            if index is None:
                attrs = {"username": row[1], "network_type": network_name}
            else:
                attrs = {"username": row[1], "network_type": network_name, "window_index": index}
            for i, message in enumerate(nodes_dict[user_id]):
                attrs[f"message_{i}"] = message
            g.add_node(user_id, **attrs)

    return g

