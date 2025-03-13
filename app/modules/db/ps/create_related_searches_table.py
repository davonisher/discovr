"""
Script to specifically create the related_searches table.
This can be run directly to ensure the table exists without requiring other tables or extensions.
"""

import os
import logging
import psycopg2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Returns a psycopg2 connection to the PostgreSQL database.
    """
    try:
        logger.info("Connecting to PostgreSQL database")
        
        # Create connection to PostgreSQL database
        conn = psycopg2.connect(
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'katse'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'discovr2')
        )
        
        return conn
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL: {e}")
        raise

def create_related_searches_table():
    """
    Creates only the related_searches table.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS related_searches (
        id SERIAL PRIMARY KEY,
        original_search_query TEXT NOT NULL,
        related_term TEXT NOT NULL,
        search_count INTEGER DEFAULT 1,
        website TEXT,
        category TEXT,
        source_file TEXT,
        inserted_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(original_search_query, related_term, website)
    );
    """
    
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
        conn.commit()
        logger.info("Successfully created or verified related_searches table")
        return True
    except Exception as e:
        logger.error(f"Failed to create related_searches table: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    logger.info("Starting creation of related_searches table")
    success = create_related_searches_table()
    if success:
        logger.info("Table creation completed successfully")
    else:
        logger.error("Table creation failed") 