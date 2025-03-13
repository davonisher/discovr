# database.py

import os
import logging
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from supabase import create_client
from config import database_url, supabase_url, supabase_key

# -------------------------------------------------------------------
# 1. Database Connections
# -------------------------------------------------------------------
def get_db_connection():
    """
    Reads environment variables for DB credentials and returns a psycopg2 connection.
    Make sure you have set DATABASE_URL in your environment.
    """
    try:
        if not database_url:
            logging.warning("DATABASE_URL environment variable not set")
            database_url = "postgresql://postgres:Liefdeplant123!@db.skyehzreswvdtbfuhupa.supabase.co:6543/postgres"

        # Log that we're connecting
        logging.info("Connecting to Supabase PostgreSQL database")

        # Create connection to Supabase PostgreSQL database
        conn = psycopg2.connect(
            database_url,
            connect_timeout=10,  # Add timeout to prevent hanging
            keepalives=1,        # Enable keepalive
            keepalives_idle=30,  # Seconds between keepalive probes
            keepalives_interval=10,  # Seconds between retries
            keepalives_count=5   # Number of retries
        )
        return conn
    except Exception as e:
        logging.error(f"Error connecting to Supabase PostgreSQL: {e}")
        raise

def get_supabase_client():
    """
    Creates and returns a Supabase client using environment variables.
    """
    try:
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
            
        supabase = create_client(supabase_url, supabase_key)
        logging.info("Successfully created Supabase client")
        return supabase
    except Exception as e:
        logging.error(f"Error creating Supabase client: {e}")
        raise

# -------------------------------------------------------------------
# 2. Table Creation / Setup
# -------------------------------------------------------------------
def create_tables_if_not_exists():
    """
    Creates the 'raw_listings' and 'enriched_listings' tables if they do not exist yet.
    Adjust column types and table structure as necessary.
    """
    create_raw_table = """
    CREATE TABLE IF NOT EXISTS raw_listings (
        id SERIAL PRIMARY KEY,
        title TEXT,
        price TEXT,
        description TEXT,
        link TEXT,
        date_col TEXT,
        usage TEXT,
        search_query TEXT,
        category TEXT,
        source_file TEXT,
        inserted_at TIMESTAMP DEFAULT NOW(),
        website TEXT,
        company TEXT
    );
    """
        
    create_enriched_table = """
    CREATE TABLE IF NOT EXISTS enriched_listings (
        id SERIAL PRIMARY KEY,
        title TEXT,
        price TEXT,
        description TEXT,
        link TEXT,
        date_col TEXT,
        usage TEXT,
        search_query TEXT,
        category TEXT,
        brand TEXT,
        model TEXT,
        classified_date TEXT,
        sentiment TEXT,
        sentiment_score NUMERIC,
        source_file TEXT,
        inserted_at TIMESTAMP DEFAULT NOW(),
        
    );
    """
        
    conn = None
    try:
        conn = get_db_connection()  # Connect to the DB
        with conn.cursor() as cur:
            cur.execute(create_raw_table)
            cur.execute(create_enriched_table)
        conn.commit()
        logging.info("Tables created or verified.")
    except Exception as e:
        logging.error(f"Failed to create tables: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def check_file_processed(filename: str, table: str) -> bool:
    """
    Check if a file has already been processed and stored in the database.
    
    Args:
        filename: Name of the file to check
        table: Table name to check ('raw_listings' or 'enriched_listings')
    
    Returns:
        bool: True if file exists in database, False otherwise
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE source_file = %s", (filename,))
            count = cur.fetchone()[0]
            return count > 0
    except Exception as e:
        logging.error(f"Error checking processed file: {e}")
        return False
    finally:
        if conn:
            conn.close()

# -------------------------------------------------------------------
# 3. Store Raw Data
# -------------------------------------------------------------------
def store_raw_data(df: pd.DataFrame, source_file: str = None):
    """
    Inserts rows from the 'df' (raw scraped data) into the raw_listings table.
    """
    if df.empty:
        logging.warning("DataFrame is empty. Nothing to store in raw_listings.")
        return

    if source_file and check_file_processed(source_file, 'raw_listings'):
        logging.info(f"File {source_file} has already been processed. Skipping.")
        return

    # We'll rename some columns if needed for matching DB columns
    expected_cols = ["Title", "Price", "Description", "Link", "Date", "Usage", "search_query", "category"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logging.warning(f"Missing columns {missing}. DataFrame may not match expected schema.")
        
    insert_sql = """
        INSERT INTO raw_listings (title, price, description, link, date_col, usage, search_query, category, source_file)
        VALUES %s
    """
            
    # Prepare rows for insertion
    rows = []
    for _, row in df.iterrows():
        rows.append([
            row.get("Title", ""),
            row.get("Price", ""),
            row.get("Description", ""),
            row.get("Link", ""),
            row.get("Date", ""),
            row.get("Usage", ""),
            row.get("search_query", None),
            row.get("category", None),
            row.get("source_file", None)
        ])
    
    logging.info(f"Preparing to insert {len(rows)} rows with source_file: {source_file}")
            
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, rows)
        conn.commit()
        logging.info(f"Successfully inserted {len(rows)} rows into raw_listings with source_file: {source_file}")
    except Exception as e:
        logging.error(f"Failed to insert into raw_listings: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# -------------------------------------------------------------------
# 3. Store Preprocessed Data
# -------------------------------------------------------------------
def store_preprocessed_data(df: pd.DataFrame, source_file: str = None):
    """
    Inserts rows from the preprocessed DataFrame into the preprocessed_listings table.
    Includes cleaned and standardized data fields.
    """
    if df.empty:
        logging.warning("DataFrame is empty. Nothing to store in preprocessed_listings.")
        return

    if source_file and check_file_processed(source_file, 'preprocessed_listings'):
        logging.info(f"File {source_file} has already been processed. Skipping.")
        return

    # Create preprocessed_listings table if it doesn't exist
    create_preprocessed_table = """
    CREATE TABLE IF NOT EXISTS preprocessed_listings (
        id SERIAL PRIMARY KEY,
        title TEXT,
        cleaned_price NUMERIC,
        link TEXT,
        date_col TEXT,
        usage TEXT,
        inserted_at TIMESTAMP DEFAULT NOW()
    );
    """

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(create_preprocessed_table)
            conn.commit()

        # Prepare insert SQL
        insert_sql = """
            INSERT INTO preprocessed_listings 
            (title, price, cleaned_price, description, link, date_col, usage, search_query, category)
            VALUES %s
        """

        # Prepare rows for insertion
        rows = []
        for _, row in df.iterrows():
            rows.append([
                row.get("Title", ""),
                row.get("Price", ""),
                row.get("CleanedPrice", None),
                row.get("Description", ""),
                row.get("Link", ""),
                row.get("Date", ""),
                row.get("Usage", ""),
                row.get("search_query", None),
                row.get("category", None),
            ])
        
        logging.info(f"Preparing to insert {len(rows)} preprocessed rows with source_file: {source_file}")
            
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, rows)
        conn.commit()
        logging.info(f"Successfully inserted {len(rows)} rows into preprocessed_listings")
            
    except Exception as e:
        logging.error(f"Failed to insert into preprocessed_listings: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# -------------------------------------------------------------------
# 4. Store Enriched Data
# -------------------------------------------------------------------
def store_enriched_data(df: pd.DataFrame, source_file: str = None):
    """
    Inserts rows from the enriched DataFrame into the enriched_listings table.
    Expects brand, model, classified_date, sentiment, sentiment_score, etc.
    """
    if df.empty:
        logging.warning("DataFrame is empty. Nothing to store in enriched_listings.")
        return

    if source_file and check_file_processed(source_file, 'enriched_listings'):
        logging.info(f"File {source_file} has already been processed. Skipping.")
        return

    expected_cols = ["Title", "Price", "Description", "Link", "Date", "Usage",
                     "Brand", "Model", "ClassifiedDate", "Sentiment", "SentimentScore"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logging.warning(f"Missing columns {missing}. DataFrame may not match expected schema.")

    insert_sql = """
        INSERT INTO enriched_listings
        (title, price, description, link, date_col, usage, search_query, category, brand, model, classified_date, sentiment, sentiment_score, source_file)
        VALUES %s
    """
        
    # Reorder columns and add search_query, category and source_file to each row
    rows = []
    for _, row in df.iterrows():
        rows.append([
            row.get("Title", None),
            row.get("Price", None), 
            row.get("Description", None),
            row.get("Link", None),
            row.get("Date", None),
            row.get("Usage", None),
            row.get("search_query", None),
            row.get("category", None),
            row.get("Brand", None),
            row.get("Model", None),
            row.get("ClassifiedDate", None),
            row.get("Sentiment", None),
            row.get("SentimentScore", None),
            source_file
        ])
                
    logging.info(f"Preparing to insert {len(rows)} rows with source_file: {source_file}")
            
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, rows)
        conn.commit()
        logging.info(f"Successfully inserted {len(rows)} rows into enriched_listings with source_file: {source_file}")
    except Exception as e:
        logging.error(f"Failed to insert into enriched_listings: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
