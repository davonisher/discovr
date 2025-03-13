"""
Example usage of the redundant database storage system.
This script demonstrates how to use the system to save data to both
PostgreSQL and Supabase databases.
"""

import os
import sys
import logging
import pandas as pd

# Set up path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the database functions
from modules.db import (
    add_raw_data_to_db,
    add_enriched_data_to_db,
    add_preprocessed_data_to_db,
    get_latest_listings,
    save_raw_data_redundant,
    save_enriched_data_redundant
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def example_save_raw_data():
    """Example of saving raw data to both databases."""
    # Create a sample DataFrame with raw listings data
    raw_data = {
        "Title": ["MacBook Pro M1", "iPhone 15 Pro", "iPad Pro 2023"],
        "Price": ["€1299", "€999", "€899"],
        "Description": [
            "Brand new MacBook Pro with M1 chip",
            "Latest iPhone model",
            "Latest iPad Pro model"
        ],
        "Link": [
            "https://example.com/macbook",
            "https://example.com/iphone",
            "https://example.com/ipad"
        ],
        "Date": ["2024-06-24", "2024-06-24", "2024-06-24"],
        "Usage": ["New", "New", "New"],
        "search_query": ["laptop", "smartphone", "tablet"],
        "category": ["Electronics", "Electronics", "Electronics"],
        "website": ["example.com", "example.com", "example.com"],
        "company": ["Apple", "Apple", "Apple"]
    }
    
    df = pd.DataFrame(raw_data)
    
    # Save the data to both databases
    logging.info("Saving raw data to both databases...")
    result = save_raw_data_redundant(df, source_file="example_raw.csv")
    
    # Check the result
    if result["overall_success"]:
        logging.info("Successfully saved raw data to at least one database")
        
        if result["postgres"]["success"]:
            logging.info("Raw data saved to PostgreSQL")
        else:
            logging.warning("Failed to save raw data to PostgreSQL")
            
        if result["supabase"]["success"]:
            logging.info("Raw data saved to Supabase")
        else:
            logging.warning("Failed to save raw data to Supabase")
    else:
        logging.error("Failed to save raw data to any database")
        
def example_save_enriched_data():
    """Example of saving enriched data to both databases."""
    # Create a sample DataFrame with enriched listings data
    enriched_data = {
        "Title": ["MacBook Pro M1", "iPhone 15 Pro", "iPad Pro 2023"],
        "Price": ["€1299", "€999", "€899"],
        "Description": [
            "Brand new MacBook Pro with M1 chip",
            "Latest iPhone model",
            "Latest iPad Pro model"
        ],
        "Link": [
            "https://example.com/macbook",
            "https://example.com/iphone",
            "https://example.com/ipad"
        ],
        "Date": ["2024-06-24", "2024-06-24", "2024-06-24"],
        "Usage": ["New", "New", "New"],
        "Brand": ["Apple", "Apple", "Apple"],
        "Model": ["MacBook Pro", "iPhone 15 Pro", "iPad Pro"],
        "ClassifiedDate": ["2024-06-24", "2024-06-24", "2024-06-24"],
        "Sentiment": ["positive", "positive", "positive"],
        "SentimentScore": [0.8, 0.9, 0.85],
        "search_query": ["laptop", "smartphone", "tablet"],
        "category": ["Electronics", "Electronics", "Electronics"],
        "website": ["example.com", "example.com", "example.com"],
        "company": ["Apple", "Apple", "Apple"]
    }
    
    df = pd.DataFrame(enriched_data)
    
    # Save the data to both databases
    logging.info("Saving enriched data to both databases...")
    result = save_enriched_data_redundant(df, source_file="example_enriched.csv")
    
    # Check the result
    if result["overall_success"]:
        logging.info("Successfully saved enriched data to at least one database")
        
        if result["postgres"]["success"]:
            logging.info("Enriched data saved to PostgreSQL")
        else:
            logging.warning("Failed to save enriched data to PostgreSQL")
            
        if result["supabase"]["success"]:
            logging.info("Enriched data saved to Supabase")
        else:
            logging.warning("Failed to save enriched data to Supabase")
    else:
        logging.error("Failed to save enriched data to any database")

def example_get_latest_listings():
    """Example of retrieving latest listings."""
    logging.info("Retrieving latest listings...")
    listings = get_latest_listings(limit=5)
    
    if listings is not None and not listings.empty:
        logging.info(f"Retrieved {len(listings)} latest listings")
        logging.info(f"First listing: {listings.iloc[0]['title']}")
    else:
        logging.warning("No listings found or error retrieving listings")

if __name__ == "__main__":
    # Run the examples
    example_save_raw_data()
    example_save_enriched_data()
    example_get_latest_listings() 