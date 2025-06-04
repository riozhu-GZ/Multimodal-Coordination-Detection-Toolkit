from datetime import datetime
import networkx as nx
import igraph as ig
import leidenalg
import pandas as pd
import numpy as np
from collections import Counter
import itertools
from sentence_transformers import util
from typing import List, Tuple


def compute_co_similar_network(db_path: str) -> nx.Graph:
    """
    Load the multimodal co-similar tweet network from the embedding database.
    """
    # from your_module import compute_co_similar_tweet_multimodal, load_networkx_graph
    
    compute_co_similar_tweet_multimodal(
        db_path,
        time_window=60,
        text_similarity_threshold=0.9,
        img_similarity_threshold=0.8,
        min_edge_weight=1,
        measure_type='multimodal_disjoint',
        embed_type='text_image',
        show_progress_bar=False
    )
    
    g_conetwork = load_networkx_graph(db_path, 'co_similar_multimodal',
                               sim_measure_type='multimodal_disjoint',
                               emb_type='text_image')

    return g_conetwork


def apply_leiden_community_detection(graph: nx.Graph, resolution: float = 0.1) -> dict:
    """
    Apply the Leiden algorithm for community detection.
    """
    g_ig = ig.Graph.from_networkx(graph)
    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.CPMVertexPartition,
        weights='weight',
        resolution_parameter=resolution
    )
    return {g_ig.vs[i]['_nx_name']: membership for i, membership in enumerate(partition.membership)}


def filter_graph_by_community(graph: nx.Graph, community_mapping: dict, min_size: int = 5) -> nx.Graph:
    """
    Filter the graph to include only communities with more than min_size nodes.
    """
    community_counts = Counter(community_mapping.values())
    communities_to_keep = {comm for comm, count in community_counts.items() if count > min_size}
    
    filtered_graph = nx.DiGraph()
    
    for node, attrs in graph.nodes(data=True):
        if community_mapping.get(node) in communities_to_keep:
            filtered_graph.add_node(node, **attrs)
            
    for u, v, attrs in graph.edges(data=True):
        if community_mapping.get(u) in communities_to_keep and community_mapping.get(v) in communities_to_keep:
            filtered_graph.add_edge(u, v, **attrs)
            
    return filtered_graph


import numpy as np
import pandas as pd
import networkx as nx
from sentence_transformers import util
from collections import Counter
import itertools

def find_embed_clusters(keys: list, corpus_embeddings: list, threshold: float = 0.9, embed_cluster_min_size: int = 2) -> tuple:
    """
    Perform community detection on embeddings, using embedding hashes as keys.
    """
    corpus_embeddings = np.array(corpus_embeddings)
    clusters = util.community_detection(
        corpus_embeddings,
        threshold=threshold,
        min_community_size=embed_cluster_min_size,
        batch_size=1024,
        show_progress_bar=False
    )

    labels = [0] * len(keys)
    for cluster_id, cluster in enumerate(clusters, start=1):
        for idx in cluster:
            labels[idx] = cluster_id

    cluster_dict = dict(zip(keys, labels))

    # Identify images in large clusters (>5)
    cluster_count = Counter(cluster_dict.values())
    large_clusters = {cid for cid, cnt in cluster_count.items() if cnt > 5}
    embed_in_large_clusters = [key for key, cid in cluster_dict.items() if cid in large_clusters]

    return cluster_dict, embed_in_large_clusters


def build_content_network(network: nx.Graph, tweet_df: pd.DataFrame, user_community_min_size: int = 5, embed_cluster_min_size: int = 3) -> nx.Graph:
    """
    Build an content-based network from tweet coordination edges, using hashes of embeddings as unique keys.
    """
    edges_df = pd.DataFrame(network.edges(data=True), columns=['Source', 'Target', 'Attributes'])
    edges_df = pd.concat([edges_df, pd.json_normalize(edges_df['Attributes'])], axis=1).drop(columns='Attributes')

    def parse_edge_pairs(pairs: str) -> list:
        return [list(map(int, p.split('-'))) for p in pairs.split(',')]

    edges_df['edges_message'] = edges_df['edges_message'].apply(parse_edge_pairs)
    edges_df = edges_df.explode('edges_message')
    edge_pairs = edges_df['edges_message'].tolist()

    tweet_ids = list(set(itertools.chain(*edge_pairs)))
    content_df = tweet_df[tweet_df['id'].isin(tweet_ids)].drop_duplicates(subset='id')

    # Compute hash keys for image embeddings
    content_df['combine_embed'] = content_df.apply(lambda row: np.concatenate([row['text_emb_from_multi'], row['image_embed']]), axis=1)
    content_df['embed_key'] = content_df['combine_embed'].apply(lambda emb: hash(tuple(emb)))

    # Map: message ID -> image_key
    id_to_embed_key = content_df.set_index('id')['embed_key'].to_dict()
    author_dict = content_df.set_index('id')['username'].to_dict()
    embeddings = content_df['combine_embed'].tolist()
    embed_keys = content_df['embed_key'].tolist()

    # Cluster images using the hashed keys
    cluster_dict, large_embed = find_embed_clusters(embed_keys, embeddings, embed_cluster_min_size)

    # Use image_keys instead of IDs for edges
    embed_pairs = [(id_to_embed_key[pair[0]], id_to_embed_key[pair[1]]) for pair in edge_pairs]
    pair_counts = Counter(embed_pairs)

    g_content = nx.Graph()
    for (src_key, tgt_key), weight in pair_counts.items():
        g_content.add_edge(src_key, tgt_key, weight=weight)

    # Aggregate attributes for each image node
    for embed_key in set(embed_keys):
        linked_ids = [str(msg_id) for msg_id, key in id_to_embed_key.items() if key == embed_key]
        usernames = [author_dict[int(msg_id)] for msg_id in linked_ids]
        attrs = {
            'message_id': ','.join(linked_ids),
            'usernames': ','.join(usernames),
            'user_count': len(usernames),
            'embed_cluster_idx': cluster_dict.get(embed_key, 0)
        }
        g_content.nodes[embed_key].update(attrs)

    # Identify large connected components
    def find_large_connected_components(graph, user_community_min_size):
        connected_components = list(nx.connected_components(graph))
        large_components = [c for c in connected_components if len(c) > user_community_min_size]
        nodes_in_large_comp = [node for comp in large_components for node in comp]
        return large_components, nodes_in_large_comp

    large_components, nodes_in_large_comp = find_large_connected_components(g_content, user_community_min_size=user_community_min_size)

    # Build final filtered graph
    g_content_big = nx.Graph()
    for u, v, data in g_content.edges(data=True):
        if u in nodes_in_large_comp and v in nodes_in_large_comp:
            g_content_big.add_edge(u, v, **data)

    for embed_key in g_content_big.nodes:
        linked_ids = [str(msg_id) for msg_id, key in id_to_embed_key.items() if key == embed_key]
        usernames = [author_dict[int(msg_id)] for msg_id in linked_ids]
        attrs = {
            'message_id': ','.join(linked_ids),
            'usernames': ','.join(usernames),
            'user_count': len(usernames),
            'embed_cluster_idx': cluster_dict.get(embed_key, 0)
        }
        g_content_big.nodes[embed_key].update(attrs)

    return g_content, g_content_big



def build_fully_connected_user_network(graph: nx.Graph) -> Tuple[List[nx.Graph], nx.Graph]:
    """
    Given a multimodal graph, create fully connected user networks
    for each image cluster based on usernames.

    Parameters:
    - graph: nx.Graph, multimodal graph with nodes containing 'image_cluster_idx' and 'usernames' attributes.

    Returns:
    - graphs: List of nx.Graph, each fully connected network for a cluster.
    - combined_graph: nx.Graph, the overall merged fully connected network.
    """
    # Create DataFrame from graph nodes
    nodes_df = pd.DataFrame(graph.nodes(data=True), columns=['Node', 'Attributes'])
    nodes_df = pd.concat([nodes_df, pd.json_normalize(nodes_df['Attributes'])], axis=1).drop(columns='Attributes')

    # Helper function to flatten comma-separated username lists
    def flatten_list(nested_list: List[str]) -> List[str]:
        flat_list = []
        for item in nested_list:
            flat_list.extend(item.split(','))
        return flat_list

    # Group users by image cluster
    nodes_group = []
    for cluster in nodes_df['embed_cluster_idx'].unique():
        users_in_cluster = nodes_df[nodes_df['embed_cluster_idx'] == cluster]['usernames'].tolist()
        users_flat = flatten_list(users_in_cluster)
        # print(f"Users in cluster {cluster}: {users_flat}")
        nodes_group.append(users_flat)

    # Create fully connected subgraphs for each group and combine them
    combined_graph = nx.Graph()

    for nodes in nodes_group:
        G = nx.complete_graph(nodes)
        combined_graph = nx.compose(combined_graph, G)

    return combined_graph


def convert_id_to_username_network(original_network, network_type = 'direct'):
    # Create a new graph for the username-based network
    if network_type == 'direct':
        username_network = nx.DiGraph()
    elif network_type == 'undirect':
        username_network = nx.Graph()

    # Iterate through each node in the original network
    for id, data in original_network.nodes(data=True):
        username = data.get('username')  # Get the username from the node attribute
        if username:
            # Add the username as a new node in the new network
            username_network.add_node(username, **data)  # Copy attributes if needed

    # Iterate through the edges in the original network
    for u, v, data in original_network.edges(data=True):
        # Get the usernames corresponding to the IDs
        username_u = original_network.nodes[u].get('username')
        username_v = original_network.nodes[v].get('username')

        # Check if both usernames exist
        if username_u and username_v:
            # Add the edge between the usernames in the new network
            username_network.add_edge(username_u, username_v, **data)

    return username_network

def compose_directed_networks(network_a, network_b):
    # Create a new directed graph for the composed network
    composed_network = nx.DiGraph()

    # Add nodes from network B to the composed network
    composed_network.add_nodes_from(network_b.nodes(data=True))


    # Iterate through edges in network A
    for u, v, data in network_a.edges(data=True):
        if u in network_b and v in network_b:
            # If both nodes exist in network B, add the directed edge to the composed network
            composed_network.add_edge(u, v, **data)  # Copy edge attributes

    for node, data in composed_network.nodes(data=True):
        composed_network.nodes[node].update(network_a.nodes[node])

    return composed_network



def compute_TiCNet():
    embedding_db_path = '/content/toy_dataset_for_test.db'
    graph_output_path = '/content/TiCNet.graphml'
    dataframe_path = '/content/toy_dataset_for_test.csv'

    print('==='*10)

    start = datetime.now()
    print("Starting multimodal TiCNet computation...")

    g_conetwork = compute_co_similar_network(embedding_db_path)
    print(f"Original network: {g_conetwork.number_of_nodes()} nodes, {g_conetwork.number_of_edges()} edges")

    community_mapping = apply_leiden_community_detection(g_conetwork)
    nx.set_node_attributes(g_conetwork, community_mapping, 'community')

    filtered_user_graph = filter_graph_by_community(g_conetwork, community_mapping, min_size=3)
    print(f"Filtered user graph: {filtered_user_graph.number_of_nodes()} nodes, {filtered_user_graph.number_of_edges()} edges")

    dataset_embed = load_and_prepare_dataframe(dataframe_path)

    g_content, g_content_big = build_content_network(filtered_user_graph, dataset_embed)

    print(f"Filtered content graph: {g_content_big.number_of_nodes()} nodes, {g_content_big.number_of_edges()} edges")

    new_user_network = build_fully_connected_user_network(g_content_big)

    g_conetwork_username = convert_id_to_username_network(filtered_user_graph, 'undirect')
    new_user_network = compose_directed_networks(g_conetwork_username, new_user_network)

    print(f"TiCNet: {new_user_network.number_of_nodes()} nodes, {new_user_network.number_of_edges()} edges")

    print("Saving filtered graph...")
    nx.write_graphml(new_user_network, graph_output_path)

    end = datetime.now()
    print(f"Processing completed in {(end - start).total_seconds() / 60:.2f} minutes")

    print('==='*10)


def compute_LiCNet():

    print('==='*10)
    
    embedding_database_path = '/content/toy_dataset_for_test.db'
    graph_output_path = '/content/LiCNet.graphml'

    start = datetime.now()
    print("Starting multimodal LiCNet computation...")

    compute_co_similar_tweet_multimodal(
        embedding_database_path,
        time_window=60*60,
        asy_min_time_window=0,
        text_similarity_threshold = .9,
        img_similarity_threshold = .8,
        min_edge_weight=5,
        measure_type = 'multimodal_disjoint',
        embed_type = 'text_image',
        show_progress_bar=False
    )

    g_conetwork = load_networkx_graph(embedding_database_path, 'co_similar_multimodal', sim_measure_type = 'multimodal_disjoint', emb_type = 'text_image')

    end = datetime.now()

    print(f"LiCNet: {g_conetwork.number_of_nodes()} nodes, {g_conetwork.number_of_edges()} edges")

    nx.write_graphml(g_conetwork, graph_output_path)

    end = datetime.now()
    print(f"Processing completed in {(end - start).total_seconds() / 60:.2f} minutes")

    print('==='*10)



def main():

    compute_TiCNet()
    compute_LiCNet()


if __name__ == '__main__':
    main()

