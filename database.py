from typing import List
import sqlite3 as lite
import pandas as pd
import numpy as np
import re
from ast import literal_eval


def initialise_multimodal_db(db_path: str) -> None:
    """
    Initialise the database schema, ensuring the correct structure for multimodal data.
    """
    db = lite.connect(db_path, isolation_level=None)
    db.executescript('''
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=normal;

        CREATE TABLE IF NOT EXISTS edge (
            message_id PRIMARY KEY,
            user_id NOT NULL,
            username TEXT,
            repost_id,
            reply_id,
            text_embed,
            image_embed,
            transformed_message TEXT
        );
    ''')
    db.close()


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
    df['text_embed'] = df['text_emb_from_multi'].apply(fix_array_string).apply(literal_eval).apply(np.array)
    df['image_embed'] = df['image_embed'].apply(fix_array_string).apply(literal_eval).apply(np.array)
    return df


def generate_combined_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Generate combined embeddings by concatenating text and image embeddings.
    """
    combined = [
        np.concatenate([text_emb, image_emb])
        for text_emb, image_emb in zip(df['text_embed'], df['image_embed'])
    ]
    return np.vstack(combined)


def insert_into_database(df: pd.DataFrame, db_path: str) -> None:
    """
    Insert processed DataFrame into SQLite database.
    """
    db = lite.connect(db_path, isolation_level=None)
    for idx, row in df.iterrows():
        db.execute('''
            INSERT OR REPLACE INTO edge (message_id, user_id, username, repost_id, reply_id, text_embed, image_embed, transformed_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                row['id'],
                row['author_id'],
                row['username'],
                row.get('retweet_id'),
                row.get('reply_id'),
                row['text_embed'].tolist(),
                row['image_embed'].tolist(),
                row.get('text', '')
            )
        )
    db.commit()
    db.close()


# if __name__ == '__main__':
#     csv_path = 'toy_dataset_for_test.csv'
#     db_path = 'output_multimodal.db'

#     print("✅ Initialising multimodal database...")
#     initialise_multimodal_db(db_path)

#     print("🔍 Loading and preparing dataset...")
#     df = load_and_prepare_dataframe(csv_path)

#     print("🔍 Generating combined embeddings...")
#     combined_embeddings = generate_combined_embeddings(df)

#     print("💾 Inserting data into database...")
#     insert_into_database(df, db_path)

#     print("🎉 Database generation complete!")
