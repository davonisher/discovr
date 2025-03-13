import os
import logging
import sys
import pandas as pd
from typing import Optional, Dict, Any

# Fix the path to properly import from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/project/modules/db
parent_dir = os.path.dirname(current_dir)  # /path/to/project/modules
project_root = os.path.dirname(parent_dir)  # /path/to/project

# Add project root to Python path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from the updated modules
from modules.db.ps.database import create_tables_if_not_exists as create_postgres_tables
from modules.db.utils import (
    save_raw_data_redundant, 
    save_enriched_data_redundant,
    save_preprocessed_data_redundant
)

def add_raw_data_to_db(file_path: str) -> bool:
    """
    Reads a CSV file from raw_data directory and adds it to both PostgreSQL and Supabase
    databases if it hasn't been processed before.
    
    Args:
        file_path: Path to the CSV file in raw_data directory
        
    Returns:
        bool: True if data was successfully added to at least one database, False otherwise
    """
    try:
        # Convert relative path to absolute path if needed
        if not os.path.isabs(file_path):
            file_path = os.path.join(project_root, file_path.lstrip('/'))
            
        # Validate file exists
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return False
            
        # Read the CSV file
        df = pd.read_csv(file_path)
        if df.empty:
            logging.warning(f"Empty CSV file: {file_path}")
            return False

        # Validate required columns
        required_columns = ["Title", "Price", "Description", "Link", "Date", "Usage"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Create PostgreSQL tables if they don't exist
        create_postgres_tables()
        
        # Get source file name from path
        source_file = os.path.basename(file_path)
            
        # Store the data to both PostgreSQL and Supabase
        result = save_raw_data_redundant(df, source_file)
        
        if result["overall_success"]:
            logging.info(f"Successfully added {len(df)} rows from {file_path} to at least one database")
            
            # Log details of which database worked
            if result["postgres"]["success"]:
                logging.info("Data saved to PostgreSQL")
            else:
                logging.warning("Failed to save data to PostgreSQL")
                
            if result["supabase"]["success"]:
                logging.info("Data saved to Supabase")
            else:
                logging.warning("Failed to save data to Supabase")
                
            return True
        else:
            logging.error(f"Failed to add data from {file_path} to any database")
            return False
        
    except pd.errors.EmptyDataError:
        logging.error(f"Empty or invalid CSV file: {file_path}")
        return False
    except Exception as e:
        logging.error(f"Error adding raw data to database: {e}")
        return False

def add_preprocessed_data_to_db(df: pd.DataFrame, source_file: str = None) -> bool:
    """
    Adds preprocessed data to the PostgreSQL database.
    
    Args:
        df: DataFrame containing the preprocessed data
        source_file: Source file name
        
    Returns:
        bool: True if data was successfully added, False otherwise
    """
    try:
        if df.empty:
            logging.warning(f"Empty DataFrame. Nothing to add to preprocessed_listings.")
            return False
        
        # Create PostgreSQL tables if they don't exist
        create_postgres_tables()
        
        # Store the preprocessed data
        result = save_preprocessed_data_redundant(df, source_file)
        
        if result["overall_success"]:
            logging.info(f"Successfully added {len(df)} preprocessed rows")
            return True
        else:
            logging.error(f"Failed to add preprocessed data")
            return False
            
    except Exception as e:
        logging.error(f"Error adding preprocessed data to database: {e}")
        return False

def add_enriched_data_to_db(file_path: str) -> bool:
    """
    Reads an enriched CSV file and adds it to both PostgreSQL and Supabase
    databases if it hasn't been processed before.
    
    Args:
        file_path: Path to the enriched CSV file
        
    Returns:
        bool: True if data was successfully added to at least one database, False otherwise
    """
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return False
            
        # Read the CSV file
        df = pd.read_csv(file_path)
        if df.empty:
            logging.warning(f"Empty CSV file: {file_path}")
            return False

        # Validate required columns
        required_columns = [
            "Title", "Price", "Description", "Link", "Date", "Usage",
            "Brand", "Model", "ClassifiedDate", "Sentiment", "SentimentScore"
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Create PostgreSQL tables if they don't exist
        create_postgres_tables()
        
        # Get source file name from path
        source_file = os.path.basename(file_path)
        
        # Store the enriched data to both PostgreSQL and Supabase
        result = save_enriched_data_redundant(df, source_file)
        
        if result["overall_success"]:
            logging.info(f"Successfully added {len(df)} enriched rows from {file_path} to at least one database")
            
            # Log details of which database worked
            if result["postgres"]["success"]:
                logging.info("Enriched data saved to PostgreSQL")
            else:
                logging.warning("Failed to save enriched data to PostgreSQL")
                
            if result["supabase"]["success"]:
                logging.info("Enriched data saved to Supabase")
            else:
                logging.warning("Failed to save enriched data to Supabase")
                
            return True
        else:
            logging.error(f"Failed to add enriched data from {file_path} to any database")
            return False
        
    except pd.errors.EmptyDataError:
        logging.error(f"Empty or invalid CSV file: {file_path}")
        return False
    except Exception as e:
        logging.error(f"Error adding enriched data to database: {e}")
        return False

def get_latest_listings(limit: int = 10) -> Optional[pd.DataFrame]:
    """
    Retrieves the latest listings from the PostgreSQL database.
    
    Args:
        limit: Maximum number of rows to return
        
    Returns:
        Optional[pd.DataFrame]: DataFrame with the latest listings or None if error
    """
    from modules.db.ps.database import get_latest_listings as get_postgres_latest
    
    try:
        # Get latest listings from PostgreSQL
        df = get_postgres_latest(limit)
        return df
    except Exception as e:
        logging.error(f"Error retrieving latest listings: {e}")
        return None

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example usage with proper path handling
    raw_data_path = os.path.join(project_root, "data", "raw_data")
    
    # Try to add raw data
    success = add_raw_data_to_db(raw_data_path)
    if success:
        logging.info("Successfully added raw data to database")
    else:
        logging.error("Failed to add raw data to database")
    
    # Try to get latest listings
    latest = get_latest_listings()
    if latest is not None:
        logging.info(f"Retrieved {len(latest)} latest listings")