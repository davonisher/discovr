import os
import time
import json
import requests
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tiktoken
from openai import OpenAI
import asyncio
import aiohttp
import nest_asyncio
# Enable nested event loops
nest_asyncio.apply()

################################################################################
# 1) Configuration & Utility
################################################################################

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
RATE_LIMIT_REQUESTS_PER_SECOND = 100  # Max 500 requests per second
MAX_TOKENS_PER_ROW = 8190  # Limit to the first 8190 tokens per row
MAX_TOKENS_PER_MINUTE = 4500000  # Max 4 million tokens per minute

def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in a given text using tiktoken.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def get_first_n_tokens(text: str, n: int) -> str:
    """
    Returns the first n tokens of a given text using tiktoken.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:n])

async def embed_text_openai_async(session, text: str) -> np.ndarray:
    """
    Asynchronously calls the OpenAI API to get a single embedding vector.
    Returns an np.array or None if there's an error.
    """
    if not text.strip():
        return None
    
    try:
        # Get the first 8190 tokens from the text
        truncated_text = get_first_n_tokens(text, MAX_TOKENS_PER_ROW)
        
        # Check token count against rate limit
        token_count = count_tokens(truncated_text)
        if token_count > MAX_TOKENS_PER_MINUTE:
            print(f"Warning: Text exceeds token rate limit ({token_count} tokens)")
            return None
            
        async with session.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            json={"input": truncated_text, "model": EMBEDDING_MODEL}
        ) as response:
            if response.status == 200:
                data = await response.json()
                emb = data['data'][0]['embedding']
                return np.array(emb, dtype="float32")
            else:
                print(f"Error embedding text with OpenAI API: {response.status}")
                return None
    except Exception as e:
        print(f"Error embedding text with OpenAI API: {e}")
        return None

################################################################################
# 2) Loading Dataset
################################################################################

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the dataset.
    Returns a DataFrame with validated columns.
    """
    try:
        df = pd.read_csv(csv_path)
        required_columns = ["Title", "Description", "Date", "Usage", "Brand", "Model"]
        
        # Check if required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            
        # Ensure all columns are string type and handle NaN values
        for col in df.columns:
            df[col] = df[col].astype(str).replace('nan', '')
            
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

################################################################################
# 3) Generate Embeddings for Each Row
################################################################################

async def generate_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate embeddings for each row in the DataFrame.
    """
    embeddings = []
    token_counts = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding rows"):
            try:
                # Filter out empty strings and 'nan' values
                text_parts = [
                    str(value) for key, value in row.items() 
                    if value and str(value).lower() != 'nan' and key in [
                        "Title", "Description", "Date", "Usage", "Brand", "Model"
                    ]
                ]
                
                combined_text = " ".join(text_parts)
                if not combined_text.strip():
                    print(f"Warning: Empty text for row {idx}")
                    embeddings.append(None)
                    token_counts.append(0)
                    continue
                
                # Count tokens
                token_count = count_tokens(combined_text)
                token_counts.append(token_count)
                
                # Create a task for each embedding request
                task = embed_text_openai_async(session, combined_text)
                tasks.append(task)
                
                # Limit the number of concurrent requests
                if len(tasks) >= RATE_LIMIT_REQUESTS_PER_SECOND:
                    results = await asyncio.gather(*tasks)
                    embeddings.extend(results)
                    tasks = []
                    
                # Add a small delay to respect rate limits
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                embeddings.append(None)
                token_counts.append(0)
        
        # Gather any remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks)
            embeddings.extend(results)
    
    # Fill None embeddings with zeros
    embeddings = [emb if emb is not None else np.zeros(1536, dtype="float32") for emb in embeddings]
    
    df["embedding"] = embeddings
    df["token_count"] = token_counts
    return df

################################################################################
# 4) Calculate Centroids for Each Use Group
################################################################################

def calculate_centroids_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by 'use_cor', average all embeddings in that group -> centroid.
    Returns a DataFrame with columns: [use_cor, centroid]
    """
    results = []
    grouped = df.groupby("Brand")
    
    for use_value, group in grouped:
        # Collect valid embeddings
        valid_embs = [emb for emb in group["embedding"] if emb is not None and len(emb) > 0]
        if len(valid_embs) == 0:
            continue
        arr = np.vstack(valid_embs)
        centroid = arr.mean(axis=0)
        results.append({"use_cor": use_value, "centroid": centroid})
    
    centroids_df = pd.DataFrame(results)
    return centroids_df

################################################################################
# 5) (Optional) K-Means Clustering
################################################################################

def cluster_embeddings(df: pd.DataFrame, n_clusters=5) -> pd.DataFrame:
    """
    Performs K-means on the row-level embeddings (or on the centroids if you prefer).
    Returns the input DataFrame with a new column 'cluster_label'.
    """
    # Extract all embeddings in a single array
    valid_idx = df["embedding"].apply(lambda x: x is not None and len(x) > 0)
    emb_array = np.vstack(df.loc[valid_idx, "embedding"].values)
    
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(emb_array)
    labels = kmeans.labels_
    
    # Assign labels back to rows
    df.loc[valid_idx, "cluster_label"] = labels
    df.loc[~valid_idx, "cluster_label"] = -1  # or NaN for missing
    return df

################################################################################
# 6) (Optional) t-SNE Visualization
################################################################################


################################################################################
# 7) Main Execution / Example
################################################################################

if __name__ == "__main__":
    try:
        # Load the original DataFrame
        csv_path = '/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Professioneel/Coding projects/marktplaats/data/class_data/Enriched_lenovo_computersensoftware_50_1737562578.csv'
        df = load_dataset(csv_path)
        print(f"Loaded dataset with {len(df)} rows")

        # Filter rows WITH Brand values
        df_with_brand = df[~df['Brand'].isna() & (df['Brand'] != '') & (df['Brand'].str.lower() != 'nan')]
        print(f"Found {len(df_with_brand)} rows with Brand values")

        if len(df_with_brand) > 0:
            # Generate embeddings for these rows
            loop = asyncio.get_event_loop()
            df_with_brand = loop.run_until_complete(generate_embeddings(df_with_brand))
            print("Embeddings generated for rows with Brand")

            # Save the embeddings
            embeddings_path = "brand_embeddings.pkl"
            df_with_brand.to_pickle(embeddings_path)
            print(f"Saved embeddings to {embeddings_path}")

            # Calculate centroids
            print("Calculating centroids by Brand...")
            centroids_df = calculate_centroids_by_group(df_with_brand)
            print(f"Generated centroids for {len(centroids_df)} brands")

            # Save centroids
            centroids_path = "use_cor_centroids.pkl"
            centroids_df.to_pickle(centroids_path)
            print(f"Saved centroids to {centroids_path}")

            # Print sample of centroids
            print("\nSample of brand centroids:")
            for _, row in centroids_df.head().iterrows():
                print(f"Brand: {row['use_cor']}")
                print(f"Centroid shape: {row['centroid'].shape}\n")
        else:
            print("No rows with Brand values found")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
