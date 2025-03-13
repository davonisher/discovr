import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import asyncio
import os
from datetime import datetime

from modules.db.ps.database import (
    get_db_connection,
    store_embedding,
    store_centroid,
    get_centroids_by_category_type
)
from app.classification import enrich_dataframe  # LLM-based enrichment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimum number of examples needed for embedding-based classification
MIN_EXAMPLES_FOR_EMBEDDING = 50

async def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generate an embedding for a text string using an embedding model.
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        List of floats representing the embedding vector, or None if failed
    """
    try:
        from openai import OpenAI
        
        # Initialize the OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Generate embedding
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"  # or newer model like "text-embedding-3-small"
        )
        
        # Extract embedding from response
        embedding = response.data[0].embedding
        
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity as a float between -1 and 1
    """
    if not a or not b:
        return 0.0
        
    a_array = np.array(a)
    b_array = np.array(b)
    
    dot_product = np.dot(a_array, b_array)
    norm_a = np.linalg.norm(a_array)
    norm_b = np.linalg.norm(b_array)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return float(dot_product / (norm_a * norm_b))

async def classify_with_embeddings(title: str, description: str, search_query: str, category_type: str = "brand") -> Tuple[Optional[str], float]:
    """
    Classify a listing using embeddings and centroid similarity.
    
    Args:
        title: Listing title
        description: Listing description
        search_query: Search query used
        category_type: Type of category to classify (e.g., "brand", "model")
        
    Returns:
        Tuple of (predicted_category, confidence_score)
    """
    # Generate embedding for the listing
    combined_text = f"{title} {description}"
    listing_embedding = await generate_embedding(combined_text)
    
    if not listing_embedding:
        logger.warning("Failed to generate embedding for listing")
        return None, 0.0
    
    # Get centroids for the category type
    centroids = get_centroids_by_category_type(category_type, search_query)
    
    if not centroids:
        logger.warning(f"No centroids found for {category_type} with search query {search_query}")
        return None, 0.0
    
    # Calculate similarity with each centroid
    similarities = []
    for category_value, centroid, count in centroids:
        sim = cosine_similarity(listing_embedding, centroid)
        similarities.append((category_value, sim, count))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    if not similarities:
        return None, 0.0
    
    # Return top match and its confidence score
    top_match = similarities[0]
    return top_match[0], top_match[1]

async def enrich_preprocessed_listing(listing_id: int, title: str, description: str, 
                                     search_query: str, category: str) -> Dict[str, Any]:
    """
    Enrich a preprocessed listing with brand, model, etc.
    
    Args:
        listing_id: ID of the preprocessed listing
        title: Listing title from the raw listing
        description: Listing description from the raw listing
        search_query: Search query from the raw listing
        category: Category from the raw listing
        
    Returns:
        Dictionary with enriched data (only the enrichment fields, not duplicating data)
    """
    # Check if we have enough examples for embedding-based classification
    conn = None
    use_embeddings = False
    brand = None
    model = None
    sentiment = None
    sentiment_score = None
    embedding_id = None
    
    try:
        conn = get_db_connection()
        
        # Count how many enriched listings we have with a valid brand for this search query
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) 
                FROM enriched_listings e
                JOIN preprocessed_listings p ON e.preprocessed_listing_id = p.id
                JOIN raw_listings r ON p.raw_listing_id = r.id
                WHERE r.search_query = %s 
                AND e.brand IS NOT NULL 
                AND e.brand != ''
            """, (search_query,))
            
            brand_count = cur.fetchone()[0]
            
            # If we have enough examples, use embedding-based classification
            if brand_count >= MIN_EXAMPLES_FOR_EMBEDDING:
                use_embeddings = True
                logger.info(f"Using embedding-based classification for listing {listing_id} (found {brand_count} examples)")
    except Exception as e:
        logger.error(f"Error checking for embedding examples: {e}")
    finally:
        if conn:
            conn.close()
    
    # Generate embedding for the listing
    combined_text = f"{title} {description}"
    embedding = await generate_embedding(combined_text)
    
    if embedding:
        # Store the embedding
        embedding_id = store_embedding(
            listing_id=listing_id,
            table_source="preprocessed_listings",
            embedding=embedding,
            search_query=search_query
        )
        logger.info(f"Stored embedding with ID {embedding_id} for listing {listing_id}")
    
    # Choose classification method
    if use_embeddings:
        # Use embedding-based classification
        brand, brand_confidence = await classify_with_embeddings(
            title=title,
            description=description,
            search_query=search_query,
            category_type="brand"
        )
        
        model, model_confidence = await classify_with_embeddings(
            title=title,
            description=description,
            search_query=search_query,
            category_type="model"
        )
        
        logger.info(f"Embedding classification: Brand={brand} ({brand_confidence:.2f}), Model={model} ({model_confidence:.2f})")
    else:
        # Use LLM-based classification
        # Create a small dataframe with just this listing
        df = pd.DataFrame({
            "Title": [title],
            "Description": [description],
            "search_query": [search_query],
            "category": [category]
        })
        
        # Enrich using the LLM-based method
        enriched_df = enrich_dataframe(df)
        
        if not enriched_df.empty:
            brand = enriched_df.iloc[0].get("Brand", None)
            model = enriched_df.iloc[0].get("Model", None)
            sentiment = enriched_df.iloc[0].get("Sentiment", None)
            sentiment_score = enriched_df.iloc[0].get("SentimentScore", None)
            
        logger.info(f"LLM classification: Brand={brand}, Model={model}")
    
    # Return the enriched data - only the fields we need to store
    return {
        "brand": brand,
        "model": model,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "classified_date": datetime.now(),
        "embedding_id": embedding_id if embedding else None
    }

async def process_preprocessed_listings_batch(batch_size: int = 10, limit: int = 50) -> Tuple[int, int]:
    """
    Process a batch of preprocessed listings and store in enriched_listings table.
    
    Args:
        batch_size: Number of listings to process in each batch
        limit: Maximum number of listings to process in total
        
    Returns:
        Tuple of (processed_count, error_count)
    """
    conn = None
    processed_count = 0
    error_count = 0
    
    try:
        conn = get_db_connection()
        
        # Get preprocessed listings that haven't been enriched yet
        with conn.cursor() as cur:
            cur.execute("""
                SELECT p.id, r.title, r.description, r.search_query, r.category
                FROM preprocessed_listings p
                JOIN raw_listings r ON p.raw_listing_id = r.id
                LEFT JOIN enriched_listings e ON p.id = e.preprocessed_listing_id
                WHERE e.id IS NULL
                AND p.processing_status = 'processed'
                ORDER BY p.inserted_at DESC
                LIMIT %s
            """, (limit,))
            
            preprocessed_listings = cur.fetchall()
            
        if not preprocessed_listings:
            logger.info("No new preprocessed listings to enrich")
            return (0, 0)
            
        logger.info(f"Found {len(preprocessed_listings)} preprocessed listings to enrich")
        
        # Process listings in batches
        for i in range(0, len(preprocessed_listings), batch_size):
            batch = preprocessed_listings[i:i+batch_size]
            
            # Process batch concurrently
            enrichment_tasks = []
            for listing in batch:
                task = enrich_preprocessed_listing(
                    listing_id=listing[0],
                    title=listing[1],
                    description=listing[2],
                    search_query=listing[3],
                    category=listing[4]
                )
                enrichment_tasks.append(task)
            
            # Wait for all enrichment tasks to complete
            enriched_results = await asyncio.gather(*enrichment_tasks, return_exceptions=True)
            
            # Insert results into enriched_listings
            for j, result in enumerate(enriched_results):
                try:
                    if isinstance(result, Exception):
                        logger.error(f"Error enriching listing {batch[j][0]}: {result}")
                        error_count += 1
                        continue
                    
                    listing_id = batch[j][0]
                    
                    # Insert into enriched_listings
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO enriched_listings 
                            (preprocessed_listing_id, brand, model, sentiment, sentiment_score, classified_date, enrichment_status)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            listing_id,
                            result.get("brand"),
                            result.get("model"),
                            result.get("sentiment"),
                            result.get("sentiment_score"),
                            result.get("classified_date"),
                            'processed'
                        ))
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error inserting enriched data for listing {batch[j][0]}: {e}")
                    error_count += 1
            
            # Commit after each batch
            conn.commit()
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(preprocessed_listings) + batch_size - 1)//batch_size}")
            
    except Exception as e:
        logger.error(f"Error in process_preprocessed_listings_batch: {e}")
        if conn:
            conn.rollback()
        return (processed_count, error_count)
    finally:
        if conn:
            conn.close()
            
    return (processed_count, error_count)

async def update_centroids_for_search_query(search_query: str, category_type: str = "brand") -> int:
    """
    Update centroids for a specific search query and category type.
    
    Args:
        search_query: Search query to update centroids for
        category_type: Type of category (e.g., "brand", "model")
        
    Returns:
        Number of centroids updated
    """
    conn = None
    updated_count = 0
    
    try:
        conn = get_db_connection()
        
        # Get all distinct category values for this search query
        with conn.cursor() as cur:
            if category_type.lower() == "brand":
                cur.execute("""
                    SELECT DISTINCT brand 
                    FROM enriched_listings 
                    WHERE search_query = %s 
                    AND brand IS NOT NULL 
                    AND brand != ''
                """, (search_query,))
            elif category_type.lower() == "model":
                cur.execute("""
                    SELECT DISTINCT model 
                    FROM enriched_listings 
                    WHERE search_query = %s 
                    AND model IS NOT NULL 
                    AND model != ''
                """, (search_query,))
            else:
                logger.error(f"Unsupported category type: {category_type}")
                return 0
                
            category_values = [row[0] for row in cur.fetchall()]
            
        if not category_values:
            logger.info(f"No {category_type} values found for search query {search_query}")
            return 0
            
        logger.info(f"Found {len(category_values)} {category_type} values for search query {search_query}")
        
        # Process each category value
        for category_value in category_values:
            # Get all listings with this category value
            with conn.cursor() as cur:
                if category_type.lower() == "brand":
                    cur.execute("""
                        SELECT e.id, e.title, e.description, emb.embedding
                        FROM enriched_listings e
                        LEFT JOIN embeddings emb ON e.id = emb.listing_id AND emb.table_source = 'enriched_listings'
                        WHERE e.search_query = %s 
                        AND e.brand = %s
                    """, (search_query, category_value))
                else:  # model
                    cur.execute("""
                        SELECT e.id, e.title, e.description, emb.embedding
                        FROM enriched_listings e
                        LEFT JOIN embeddings emb ON e.id = emb.listing_id AND emb.table_source = 'enriched_listings'
                        WHERE e.search_query = %s 
                        AND e.model = %s
                    """, (search_query, category_value))
                    
                listings = cur.fetchall()
            
            if not listings:
                logger.warning(f"No listings found for {category_type}={category_value}")
                continue
                
            # Collect embeddings
            embeddings = []
            for listing in listings:
                if listing[3]:  # If embedding exists
                    embeddings.append(listing[3])
                else:
                    # Generate embedding for this listing
                    combined_text = f"{listing[1]} {listing[2]}"  # title + description
                    embedding = await generate_embedding(combined_text)
                    
                    if embedding:
                        # Store the embedding
                        store_embedding(
                            listing_id=listing[0],
                            table_source="enriched_listings",
                            embedding=embedding,
                            search_query=search_query,
                            brand=category_value if category_type.lower() == "brand" else None,
                            model=category_value if category_type.lower() == "model" else None
                        )
                        
                        embeddings.append(embedding)
            
            if not embeddings:
                logger.warning(f"No embeddings found for {category_type}={category_value}")
                continue
                
            # Calculate centroid (average of all embeddings)
            embeddings_array = np.array(embeddings)
            centroid = embeddings_array.mean(axis=0).tolist()
            
            # Store centroid
            centroid_id = store_centroid(
                category_type=category_type,
                category_value=category_value,
                embedding=centroid,
                count=len(embeddings),
                search_query=search_query
            )
            
            if centroid_id:
                updated_count += 1
                logger.info(f"Updated centroid for {category_type}={category_value} with ID {centroid_id}")
                
    except Exception as e:
        logger.error(f"Error updating centroids: {e}")
        if conn:
            conn.rollback()
        return updated_count
    finally:
        if conn:
            conn.close()
            
    return updated_count

async def main():
    """
    Main function to run the enrichment pipeline.
    """
    # Process a batch of preprocessed listings
    processed, errors = await process_preprocessed_listings_batch(batch_size=10, limit=50)
    print(f"Processed {processed} preprocessed listings with {errors} errors")
    
    # Update centroids for active search queries
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT search_query FROM enriched_listings LIMIT 5")
            search_queries = [row[0] for row in cur.fetchall()]
            
        for query in search_queries:
            if query:
                updated = await update_centroids_for_search_query(query, "brand")
                print(f"Updated {updated} brand centroids for query '{query}'")
                
                updated = await update_centroids_for_search_query(query, "model")
                print(f"Updated {updated} model centroids for query '{query}'")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 