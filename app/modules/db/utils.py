import logging
import pandas as pd
from typing import Dict, Any, Optional, Tuple

# Import PostgreSQL functions from new location
from modules.db.ps.database import store_raw_data, store_enriched_data, store_preprocessed_data

# Import Supabase functions
from modules.db.sb.client import (
    dataframe_to_supabase_raw,
    dataframe_to_supabase_enriched
)

def save_raw_data_redundant(df: pd.DataFrame, source_file: str = None) -> Dict[str, Any]:
    """
    Save raw listing data to both PostgreSQL and Supabase databases for redundancy.
    
    Args:
        df: DataFrame containing the raw listing data
        source_file: Source file name
        
    Returns:
        Dict containing success status and responses from both databases
    """
    result = {
        "postgres": {"success": False},
        "supabase": {"success": False},
        "overall_success": False
    }
    
    try:
        # Save to PostgreSQL
        try:
            logging.info(f"Saving {len(df)} raw listings to PostgreSQL...")
            store_raw_data(df, source_file)
            result["postgres"] = {"success": True}
            logging.info("Successfully saved raw listings to PostgreSQL")
        except Exception as e:
            logging.error(f"Error saving raw listings to PostgreSQL: {e}")
            result["postgres"] = {"success": False, "error": str(e)}
        
        # Save to Supabase
        try:
            logging.info(f"Saving {len(df)} raw listings to Supabase...")
            supabase_result = dataframe_to_supabase_raw(df, source_file)
            result["supabase"] = supabase_result
            if supabase_result["success"]:
                logging.info("Successfully saved raw listings to Supabase")
            else:
                logging.error(f"Error saving raw listings to Supabase: {supabase_result.get('error')}")
        except Exception as e:
            logging.error(f"Error saving raw listings to Supabase: {e}")
            result["supabase"] = {"success": False, "error": str(e)}
        
        # Overall success if at least one database save succeeded
        result["overall_success"] = result["postgres"]["success"] or result["supabase"]["success"]
        
        return result
    except Exception as e:
        logging.error(f"Unexpected error in save_raw_data_redundant: {e}")
        result["error"] = str(e)
        return result

def save_enriched_data_redundant(df: pd.DataFrame, source_file: str = None) -> Dict[str, Any]:
    """
    Save enriched listing data to both PostgreSQL and Supabase databases for redundancy.
    
    Args:
        df: DataFrame containing the enriched listing data
        source_file: Source file name
        
    Returns:
        Dict containing success status and responses from both databases
    """
    result = {
        "postgres": {"success": False},
        "supabase": {"success": False},
        "overall_success": False
    }
    
    try:
        # Save to PostgreSQL
        try:
            logging.info(f"Saving {len(df)} enriched listings to PostgreSQL...")
            store_enriched_data(df, source_file)
            result["postgres"] = {"success": True}
            logging.info("Successfully saved enriched listings to PostgreSQL")
        except Exception as e:
            logging.error(f"Error saving enriched listings to PostgreSQL: {e}")
            result["postgres"] = {"success": False, "error": str(e)}
        
        # Save to Supabase
        try:
            logging.info(f"Saving {len(df)} enriched listings to Supabase...")
            supabase_result = dataframe_to_supabase_enriched(df, source_file)
            result["supabase"] = supabase_result
            if supabase_result["success"]:
                logging.info("Successfully saved enriched listings to Supabase")
            else:
                logging.error(f"Error saving enriched listings to Supabase: {supabase_result.get('error')}")
        except Exception as e:
            logging.error(f"Error saving enriched listings to Supabase: {e}")
            result["supabase"] = {"success": False, "error": str(e)}
        
        # Overall success if at least one database save succeeded
        result["overall_success"] = result["postgres"]["success"] or result["supabase"]["success"]
        
        return result
    except Exception as e:
        logging.error(f"Unexpected error in save_enriched_data_redundant: {e}")
        result["error"] = str(e)
        return result

def save_preprocessed_data_redundant(df: pd.DataFrame, source_file: str = None) -> Dict[str, Any]:
    """
    Save preprocessed listing data to PostgreSQL (we only store preprocessed data in PostgreSQL for now).
    
    Args:
        df: DataFrame containing the preprocessed listing data
        source_file: Source file name
        
    Returns:
        Dict containing success status and response from PostgreSQL
    """
    result = {
        "postgres": {"success": False},
        "overall_success": False
    }
    
    try:
        # Save to PostgreSQL
        try:
            logging.info(f"Saving {len(df)} preprocessed listings to PostgreSQL...")
            store_preprocessed_data(df, source_file)
            result["postgres"] = {"success": True}
            result["overall_success"] = True
            logging.info("Successfully saved preprocessed listings to PostgreSQL")
        except Exception as e:
            logging.error(f"Error saving preprocessed listings to PostgreSQL: {e}")
            result["postgres"] = {"success": False, "error": str(e)}
            result["overall_success"] = False
        
        return result
    except Exception as e:
        logging.error(f"Unexpected error in save_preprocessed_data_redundant: {e}")
        result["error"] = str(e)
        return result 