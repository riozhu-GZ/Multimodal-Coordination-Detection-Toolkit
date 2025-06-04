# !pip install coordination-network-toolkit
# !pip install python-louvain
# !pip install leidenalg igraph python-igraph
# !pip install -U sentence-transformers

COMMAND_TABLE = {
    "co_retweet": "co_retweet_network",
    "co_tweet": "co_tweet_network",
    "co_reply": "co_reply_network",
    "co_link": "co_link_network",
    "co_similar_tweet": "co_similar_tweet_network",
    "co_post": "co_post_network",
}

COMMAND_MULTIMODAL_TABLE = {
    "co_similar_multimodal": "co_similar_multimodal_network",
    "co_similar_image":"co_similar_image",
}


from typing import List, Callable, Iterable
import sqlite3 as lite
import pandas as pd
import numpy as np
import re
from ast import literal_eval

# import coordination_network_toolkit as coord_net_tk
import networkx as nx
import concurrent.futures
from tqdm import tqdm
import os


def initialise_multimodal_db(db_path: str):
    """
    Initialise the database, ensuring the correct schema is in place.

    Raises a ValueError if the on disk format is incompatible with this version.

    """

    db = lite.connect(db_path, isolation_level=None)

    db.executescript(
        """
        pragma journal_mode=WAL;
        pragma synchronous=normal;

        create table if not exists edge (
            message_id primary key,
            user_id not null,
            username text,
            repost_id,
            reply_id,
            text_embed,
            image_embed,
            -- The following now describes the transformation of the message,
            -- as co-tweet analysis needs to be robust to some non-consequential
            -- variations.
            transformed_message text,
            transformed_message_length integer,
            transformed_message_hash blob,
            token_set text,
            timestamp integer
        );

        -- To accelerate the lookup of user_ids, and messages by specific users.
        create index if not exists user_edge on edge(user_id);

        create table if not exists message_url(
            message_id references edge(message_id),
            url,
            timestamp,
            user_id,
            primary key (message_id, url)
        );

        create table if not exists resolved_url(
            url primary key,
            resolved_url,
            ssl_verified,
            resolved_status
        );

        create trigger if not exists url_to_resolve after insert on message_url
            begin
                insert or ignore into resolved_url(url) values(new.url);
            end;

        create table if not exists metadata (
            property primary key,
            value
        );

        insert or ignore into metadata values('version', 1);
        """
    )

    # Sniff the columns in the ondisk format, to handle databases created
    # before the version check
    edge_columns = {row[1] for row in db.execute("pragma table_info('edge')")}

    # Current version in the database
    version = list(db.execute("select value from metadata where property = 'version'"))[
        0
    ][0]

    if "message_length" in edge_columns or version != 1:
        raise ValueError(
            "This database is not compatible with this version of the "
            "coordination network toolkit - you will need to reprocess your data "
            "into a new database."
        )

    return db

def preprocess_multimodal_data(db_path: str, messages: Iterable):

    db = lite.connect(db_path, isolation_level=None)

    try:
        db.execute("begin")
        processed = (
            (
                message_id,
                user_id,
                username,
                repost_id or None,
                reply_id or None,
                text_embed or None,
                image_embed or None,
                # These will be populated when co-tweet calculations are necessary
                None,
                None,
                None,
                # This will be populated only when similarity calculations are necessary
                None,
                float(timestamp),
                urls,
            )
            for message_id, user_id, username, repost_id, reply_id, text_embed, image_embed, timestamp, urls in messages
        )

        for row in processed:
            db.execute(
                "insert or ignore into edge values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row[:-1],
            )

            message_id, user_id = row[:2]
            timestamp = row[-2]

            # Ignore url shared in reposts
            if not row[3]:
                for url in row[-1]:
                    db.execute(
                        "insert or ignore into message_url values(?, ?, ?, ?)",
                        (message_id, url, timestamp, user_id),
                    )

        db.execute("commit")
    finally:
        db.close()



def divide_dataframe_into_chunk(df, window_size):
    # Divide dataframe into chunks - a overlapping window (overlap = 1/2 window)
    # and arrange regular window and overlap window seperately to avoid lock in concurrency
    df = df.sort_values(by='timestamp')
    index = 0
    chunks = []
    for i in tqdm(range(0, len(df)-window_size+1, int(window_size))):

        chunk = df.iloc[i:i + window_size].index
        chunks.append(chunk)

    return chunks



def preprocess_chunk(command, db_path, input_df, df_index):
    # Preprocess each chunk into database
    input_df.loc[:, 'text_emb_from_multi'] = input_df['text_emb_from_multi'].apply(lambda x: x.tolist())
    input_df.loc[:, 'image_embed'] = input_df['image_embed'].apply(lambda x: x.tolist())

    def list_to_str(lst):
        if not isinstance(lst, list):
            return None
        return ','.join(map(str, lst))

    input_df.loc[:, 'text_emb_from_multi'] = input_df['text_emb_from_multi'].apply(list_to_str)
    input_df.loc[:, 'image_embed'] = input_df['image_embed'].apply(list_to_str)

    data = ((*row.iloc[:-1], str(row.iloc[-1]).split(" ")) for _, row in input_df.iterrows())

    preprocess_multimodal_data(db_path, data)

    return db_path



# Check database size
def check_database_size(db_path):
    # Connect to the database
    conn = lite.connect(db_path)
    cursor = conn.cursor()

    # Execute the query to get the number of rows in edge_table
    cursor.execute("SELECT COUNT(*) FROM edge")
    row_count = cursor.fetchone()[0]

    return row_count


def watchdog_database(db_path):
    # Establish a connection to the database
    try:
        conn = lite.connect(db_path)
        cursor = conn.cursor()

        # SQL query to select two columns
        query = "SELECT text_emb, image_embed FROM edge"

        # Execute the query
        cursor.execute(query)
        results = cursor.fetchall()
    except Exception as e:
        print(e)




# An abstract pipepline for each chunk
def concurrent_pipepline(index, chunk, db_path,
                        command,
                        embed_type = 'text_image'):

      if index % 500 == 0 and index != 0:
          print('Processing network_' + str(index))

      db = initialise_multimodal_db(db_path)

      db_path = preprocess_chunk(command, db_path, chunk, index)


      size = check_database_size(db_path)
      if index % 500 == 0 and index != 0:
          print('Current database size: {}'.format(size))
          watchdog_database(db_path)

      return size



def generate_database_from_chunks(df, chunk_indices,
                                 db_path,
                                 command,
                                 embed_type = 'text_image'
                                 ):

    print('Processing embedding database: {}'.format(*[embed_type]))

    graph_lst = []
    with tqdm(total = len(chunk_indices), desc = 'Processing on chunks') as pbar:
          with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:

            for index, chunk_index in enumerate(chunk_indices):
                chunk = df.loc[chunk_index,]
                params = (index, chunk, db_path, command, embed_type)
                future = executor.submit(concurrent_pipepline, *params)

                g_co = future.result()
                graph_lst.append(g_co)
                pbar.update(1)



def fix_array_string(array_str: str) -> str:
    """
    Fixes missing commas in array-like strings for literal_eval.
    """
    inner_str = array_str.strip()[1:-1]
    inner_str = re.sub(r'\s+', ' ', inner_str)
    fixed_str = re.sub(r'(?<=[0-9e]) (?=[-]?[0-9])', ', ', inner_str)
    return f'[{fixed_str}]'


def load_and_prepare_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load dataset, fix missing commas, and convert string representations to numpy arrays.
    """
    df = pd.read_csv(csv_path)
    df['text_emb_from_multi'] = df['text_emb_from_multi'].apply(fix_array_string).apply(literal_eval).apply(np.array)
    df['image_embed'] = df['image_embed'].apply(fix_array_string).apply(literal_eval).apply(np.array)
    return df



def main():
    csv_path = 'toy_dataset_for_test.csv'
    db_path = 'toy_dataset_for_test.db'

    print("Initialising multimodal database...")
    initialise_multimodal_db(db_path)

    print("Loading and preparing dataset...")
    df = load_and_prepare_dataframe(csv_path)

    chunk_indices = divide_dataframe_into_chunk(df, window_size = 10)
    print('Number of Chunks {} in Dataset with size of {}'.format(len(chunk_indices), len(df)))

    print("Inserting data into database...")
    generate_database_from_chunks(df, chunk_indices, db_path, 'co_similar_multimodal',
                                        embed_type = 'text_image')

    print("\nDatabase generation complete!")


if __name__ == '__main__':
    main()

