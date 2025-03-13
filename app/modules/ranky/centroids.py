import pandas as pd
import numpy as np

##############################
# 1) Load & Prep Data
##############################

# Load the embeddings we just generated
df_with_embeddings = pd.read_pickle("/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Professioneel/Coding projects/marktplaats/modules/ranky/brand_embeddings.pkl")
print(f"Loaded {len(df_with_embeddings)} rows with embeddings")

# Load the centroids DataFrame (assuming you have this from previous runs)
try:
    df_use_centroids = pd.read_pickle("/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Professioneel/Coding projects/marktplaats/modules/ranky/use_cor_centroids.pkl")
    print(f"Loaded {len(df_use_centroids)} use_cor centroids")
except FileNotFoundError:
    print("Warning: use_cor_centroids.pkl not found. Please generate centroids first.")
    exit(1)

##############################
# 2) Filter Tools by token_count
##############################
df_filtered = df_with_embeddings[df_with_embeddings["token_count"] > 0].copy()
df_filtered = df_filtered.dropna(subset=["embedding"])
print(f"Working with {len(df_filtered)} valid rows after filtering")

##############################
# 3) Define Cosine Similarity
##############################
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    # ensure numpy arrays
    a = np.array(a, dtype="float32")
    b = np.array(b, dtype="float32")
    if len(a) == 0 or len(b) == 0:
        return 0.0
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

##############################
# 4) Match Each Row to Top 5 Uses
##############################
best_matches = []

for idx, row in df_filtered.iterrows():
    embedding = row["embedding"]
    
    # Store all similarities
    similarities = []
    for _, c_row in df_use_centroids.iterrows():
        centroid_vec = c_row["centroid"]
        sim = cosine_similarity(embedding, centroid_vec)
        similarities.append((c_row["use_cor"], sim))
    
    # Sort by similarity score and get top 5
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_5 = similarities[:5]
    
    # Store the results
    best_matches.append({
        "index": idx,
        "title": row.get("Title", ""),
        "brand": row.get("Brand", ""),
        "top_5_uses": [use for use, _ in top_5],
        "top_5_scores": [score for _, score in top_5]
    })

##############################
# 5) Save Results
##############################
# Convert to DataFrame
df_match_results = pd.DataFrame(best_matches)
df_match_results.set_index("index", inplace=True)

# Merge back with original data
df_final = df_filtered.merge(df_match_results[["top_5_uses", "top_5_scores"]], 
                           left_index=True, right_index=True, how="left")

# Save results
output_path = "brand_matches_with_top5_uses.csv"
df_final.to_csv(output_path, index=True)
print(f"\nResults saved to {output_path}")

# Print sample results
print("\nSample matches:")
for _, row in df_match_results.head().iterrows():
    print(f"\nTitle: {row['title']}")
    print(f"Brand: {row['brand']}")
    print("Top 5 uses and scores:")
    for use, score in zip(row['top_5_uses'], row['top_5_scores']):
        print(f"  {use}: {score:.3f}")