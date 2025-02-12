import os
import pandas as pd
import numpy as np
import asyncio
from ranky import load_dataset, generate_embeddings, cosine_similarity

################################################################################
# Main Execution
################################################################################

if __name__ == "__main__":
    try:
        # Load the original DataFrame with new items to categorize
        csv_path = '/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Professioneel/Coding projects/marktplaats/data/class_data/Enriched_lenovo_computersensoftware_50_1737562578.csv'
        df = load_dataset(csv_path)
        print(f"Loaded dataset with {len(df)} rows")

        # Load existing brand centroids
        try:
            centroids_df = pd.read_pickle("use_cor_centroids.pkl")
            print(f"Loaded {len(centroids_df)} brand centroids")
        except FileNotFoundError:
            print("Error: use_cor_centroids.pkl not found. Please run ranky.py first to generate centroids.")
            exit(1)

        # Generate embeddings for all rows
        print("Generating embeddings for new items...")
        loop = asyncio.get_event_loop()
        df_with_embeddings = loop.run_until_complete(generate_embeddings(df))
        print("Embeddings generated successfully")

        # Calculate similarities and find best matching brands
        print("Calculating brand matches...")
        matches = []
        
        for idx, row in df_with_embeddings.iterrows():
            embedding = row["embedding"]
            if embedding is None:
                continue
                
            # Calculate similarity with each brand centroid
            similarities = []
            for _, centroid_row in centroids_df.iterrows():
                brand = centroid_row["use_cor"]
                centroid = centroid_row["centroid"]
                sim = cosine_similarity(embedding, centroid)
                similarities.append((brand, sim))
            
            # Sort by similarity and get top 5
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_5 = similarities[:5]
            
            matches.append({
                "index": idx,
                "title": row.get("Title", ""),
                "current_brand": row.get("Brand", ""),
                "predicted_brands": [brand for brand, _ in top_5],
                "similarity_scores": [score for _, score in top_5]
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(matches)
        results_df.set_index("index", inplace=True)

        # Save results
        output_path = "new_items_brand_predictions.csv"
        results_df.to_csv(output_path)
        print(f"\nResults saved to {output_path}")

        # Print sample results
        print("\nSample predictions:")
        for _, row in results_df.head().iterrows():
            print(f"\nTitle: {row['title']}")
            print(f"Current Brand: {row['current_brand']}")
            print("Top 5 predicted brands and scores:")
            for brand, score in zip(row['predicted_brands'], row['similarity_scores']):
                print(f"  {brand}: {score:.3f}")

    except Exception as e:
        print(f"Error in main execution: {e}")
