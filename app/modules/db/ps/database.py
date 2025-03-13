import os
import logging
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd

# Import database URL from config
from config import database_url
#import


def get_db_connection():
    """
    Returns a psycopg2 connection to the PostgreSQL database.
    """
    try:
        # Log that we're connecting
        logging.info("Connecting to PostgreSQL database")

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
        logging.error(f"Error connecting to PostgreSQL: {e}")
        raise

def create_tables_if_not_exists():
    """
    Creates the necessary tables if they do not exist yet.
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
        preprocessed_listing_id INTEGER UNIQUE NOT NULL,
        brand TEXT,
        model TEXT,
        sentiment TEXT,
        sentiment_score NUMERIC,
        classified_date TIMESTAMP,
        inserted_at TIMESTAMP DEFAULT NOW(),
        enrichment_status TEXT DEFAULT 'pending',
        CONSTRAINT fk_preprocessed_listing
            FOREIGN KEY(preprocessed_listing_id)
            REFERENCES preprocessed_listings(id)
            ON DELETE CASCADE
    );
    """
    
    create_preprocessed_table = """
    CREATE TABLE IF NOT EXISTS preprocessed_listings (
        id SERIAL PRIMARY KEY,
        raw_listing_id INTEGER UNIQUE NOT NULL,
        cleaned_price NUMERIC,
        parsed_date TIMESTAMP,
        processing_status TEXT DEFAULT 'pending',
        inserted_at TIMESTAMP DEFAULT NOW(),
        additional_data1 TEXT,
        additional_data2 TEXT,
        CONSTRAINT fk_raw_listing
            FOREIGN KEY(raw_listing_id)
            REFERENCES raw_listings(id)
            ON DELETE CASCADE
    );
    """
    
    # Table that requires pgvector extension
    create_embeddings_table = """
    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        listing_id INTEGER NOT NULL,
        table_source TEXT NOT NULL,
        embedding VECTOR(1536),
        search_query TEXT,
        brand TEXT,
        model TEXT,
        inserted_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT fk_listing_raw
            FOREIGN KEY(listing_id)
            REFERENCES raw_listings(id)
            ON DELETE CASCADE
    );
    """
    
    # Table that requires pgvector extension
    create_centroids_table = """
    CREATE TABLE IF NOT EXISTS centroids (
        id SERIAL PRIMARY KEY,
        category_type TEXT NOT NULL,
        category_value TEXT NOT NULL,
        embedding VECTOR(1536) NOT NULL,
        count INTEGER DEFAULT 0,
        search_query TEXT,
        inserted_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(category_type, category_value, search_query)
    );
    """
    
    create_related_searches_table = """
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
            # First, create the basic tables that don't require extensions
            cur.execute(create_raw_table)
            conn.commit()
            logging.info("Created or verified raw_listings table")
            
            cur.execute(create_enriched_table)
            conn.commit()
            logging.info("Created or verified enriched_listings table")
            
            cur.execute(create_preprocessed_table)
            conn.commit()
            logging.info("Created or verified preprocessed_listings table")
            
            cur.execute(create_related_searches_table)
            conn.commit()
            logging.info("Created or verified related_searches table")
            
            # Try to create pgvector extension and tables, but don't fail if the extension is not available
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                logging.info("Created or verified vector extension")
                
                # Create the tables that depend on vector extension
                cur.execute(create_embeddings_table)
                conn.commit()
                logging.info("Created or verified embeddings table")
                
                cur.execute(create_centroids_table)
                conn.commit()
                logging.info("Created or verified centroids table")
            except Exception as vector_error:
                logging.warning(f"Could not create pgvector-dependent tables: {vector_error}")
                logging.warning("Vector embedding features will not be available. Install the pgvector extension to enable them.")
                conn.rollback()  # Roll back the failed vector extension creation, but continue with other tables
        
        logging.info("All required tables created or verified.")
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

def store_raw_data(df: pd.DataFrame, source_file: str = None):
    """
    Inserts rows from the DataFrame into the raw_listings table.
    
    Args:
        df: DataFrame with raw listing data
        source_file: Source file name
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
        INSERT INTO raw_listings 
        (title, price, description, link, date_col, usage, search_query, category, source_file, website, company)
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
            source_file or row.get("source_file", None),
            row.get("website", None),
            row.get("company", None)
        ])
    
    logging.info(f"Preparing to insert {len(rows)} rows with source_file: {source_file}")
            
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, rows)
        conn.commit()
        logging.info(f"Successfully inserted {len(rows)} rows into raw_listings")
    except Exception as e:
        logging.error(f"Failed to insert into raw_listings: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def store_preprocessed_data(df: pd.DataFrame, source_file: str = None):
    """
    Inserts rows from the preprocessed DataFrame into the preprocessed_listings table.
    
    Args:
        df: DataFrame with preprocessed listing data
        source_file: Source file name
    """
    if df.empty:
        logging.warning("DataFrame is empty. Nothing to store in preprocessed_listings.")
        return

    if source_file and check_file_processed(source_file, 'preprocessed_listings'):
        logging.info(f"File {source_file} has already been processed. Skipping.")
        return

    # Prepare insert SQL
    insert_sql = """
        INSERT INTO preprocessed_listings 
        (raw_listing_id, cleaned_price, parsed_date, processing_status)
        VALUES %s
        ON CONFLICT (raw_listing_id) 
        DO UPDATE SET 
            cleaned_price = EXCLUDED.cleaned_price,
            parsed_date = EXCLUDED.parsed_date,
            processing_status = 'processed'
    """

    # Prepare rows for insertion
    rows = []
    for _, row in df.iterrows():
        # Get the raw_listing_id (required)
        raw_listing_id = row.get("raw_listing_id")
        if not raw_listing_id:
            logging.warning(f"Skipping row without raw_listing_id: {row}")
            continue
            
        rows.append([
            raw_listing_id,
            row.get("CleanedPrice", None),
            row.get("ParsedDate", None),
            'processed'
        ])
    
    logging.info(f"Preparing to insert {len(rows)} preprocessed rows")
        
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, rows)
        conn.commit()
        logging.info(f"Successfully inserted {len(rows)} rows into preprocessed_listings")
    except Exception as e:
        logging.error(f"Failed to insert into preprocessed_listings: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def store_enriched_data(df: pd.DataFrame, source_file: str = None):
    """
    Inserts rows from the enriched DataFrame into the enriched_listings table.
    
    Args:
        df: DataFrame with enriched listing data
        source_file: Source file name
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
        (preprocessed_listing_id, brand, model, sentiment, sentiment_score, classified_date)
        VALUES %s
    """
        
    # Reorder columns and add search_query, category and source_file to each row
    rows = []
    for _, row in df.iterrows():
        rows.append([
            row.get("preprocessed_listing_id", None),
            row.get("Brand", None),
            row.get("Model", None),
            row.get("Sentiment", None),
            row.get("SentimentScore", None),
            row.get("ClassifiedDate", None)
        ])
                
    logging.info(f"Preparing to insert {len(rows)} rows with source_file: {source_file}")
            
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, rows)
        conn.commit()
        logging.info(f"Successfully inserted {len(rows)} rows into enriched_listings")
    except Exception as e:
        logging.error(f"Failed to insert into enriched_listings: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def get_latest_listings(limit: int = 10):
    """
    Retrieves the latest listings from the database.
    
    Args:
        limit: Maximum number of rows to return
        
    Returns:
        DataFrame with the latest listings
    """
    conn = None
    try:
        # Create tables if they don't exist
        create_tables_if_not_exists()
        
        conn = get_db_connection()
        query = f"""
            SELECT * FROM raw_listings 
            ORDER BY inserted_at DESC 
            LIMIT {limit}
        """
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        logging.error(f"Error retrieving latest listings: {e}")
        return None
    finally:
        if conn:
            conn.close() 

def store_embedding(listing_id: int, table_source: str, embedding: list, search_query: str = None, brand: str = None, model: str = None):
    """
    Store an embedding vector for a listing in the embeddings table.
    
    Args:
        listing_id: ID of the related listing
        table_source: Source table (e.g., 'raw_listings', 'enriched_listings')
        embedding: Vector embedding as a list of floats
        search_query: The search query used
        brand: Brand classification if available
        model: Model classification if available
    """
    insert_sql = """
        INSERT INTO embeddings 
        (listing_id, table_source, embedding, search_query, brand, model)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(insert_sql, (listing_id, table_source, embedding, search_query, brand, model))
            embedding_id = cur.fetchone()[0]
        conn.commit()
        logging.info(f"Successfully stored embedding with ID {embedding_id}")
        return embedding_id
    except Exception as e:
        logging.error(f"Failed to store embedding: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()

def store_centroid(category_type: str, category_value: str, embedding: list, count: int, search_query: str = None):
    """
    Store or update a centroid for a specific category.
    
    Args:
        category_type: Type of category (e.g., 'brand', 'model')
        category_value: Value of the category (e.g., 'Apple', 'iPhone')
        embedding: Centroid vector as a list of floats
        count: Number of embeddings used to compute this centroid
        search_query: The search query if relevant
    """
    upsert_sql = """
        INSERT INTO centroids 
        (category_type, category_value, embedding, count, search_query, updated_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (category_type, category_value, search_query)
        DO UPDATE SET 
            embedding = EXCLUDED.embedding,
            count = EXCLUDED.count,
            updated_at = NOW()
        RETURNING id
    """
    
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(upsert_sql, (category_type, category_value, embedding, count, search_query))
            centroid_id = cur.fetchone()[0]
        conn.commit()
        logging.info(f"Successfully stored/updated centroid for {category_type}:{category_value} with ID {centroid_id}")
        return centroid_id
    except Exception as e:
        logging.error(f"Failed to store centroid: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()

def get_centroids_by_category_type(category_type: str, search_query: str = None):
    """
    Retrieve all centroids for a specific category type.
    
    Args:
        category_type: Type of category (e.g., 'brand', 'model')
        search_query: Optional filter by search query
        
    Returns:
        List of tuples (category_value, embedding, count)
    """
    query = """
        SELECT category_value, embedding, count
        FROM centroids
        WHERE category_type = %s
    """
    
    params = [category_type]
    
    if search_query:
        query += " AND search_query = %s"
        params.append(search_query)
        
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
        return results
    except Exception as e:
        logging.error(f"Failed to retrieve centroids: {e}")
        return []
    finally:
        if conn:
            conn.close() 

def store_related_searches(related_df: pd.DataFrame, original_search_query: str, website: str, category: str = None, source_file: str = None):
    """
    Store related searches in the related_searches table.
    
    Args:
        related_df: DataFrame with related search terms
        original_search_query: The original search query that yielded these related terms
        website: The website these related searches came from
        category: The category of the search (optional)
        source_file: The source file name (optional)
    
    Returns:
        Dictionary with success status and info
    """
    if related_df.empty:
        logging.warning("Related searches DataFrame is empty. Nothing to store.")
        return {"success": False, "error": "Empty DataFrame"}
    
    # We'll make some assumptions about the structure of related_df:
    # - If there's a column named 'term', 'related_term', or 'search_term', we'll use that
    # - Otherwise we'll use the first column
    
    term_col = None
    for possible_col in ['term', 'related_term', 'search_term', 'query', 'related_query']:
        if possible_col in related_df.columns:
            term_col = possible_col
            break
    
    if term_col is None:
        # Use the first column as the term
        term_col = related_df.columns[0]
        logging.info(f"No standard term column found, using first column: {term_col}")
    
    # Prepare for upsert operation
    upsert_sql = """
        INSERT INTO related_searches 
        (original_search_query, related_term, website, category, source_file)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (original_search_query, related_term, website) 
        DO UPDATE SET 
            search_count = related_searches.search_count + 1,
            inserted_at = NOW()
    """
    
    inserted_count = 0
    conn = None
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Insert each related term
            for _, row in related_df.iterrows():
                related_term = row[term_col]
                if not related_term or pd.isna(related_term):
                    continue
                    
                cur.execute(upsert_sql, (
                    original_search_query,
                    related_term,
                    website,
                    category,
                    source_file
                ))
                inserted_count += 1
                
        conn.commit()
        logging.info(f"Successfully stored {inserted_count} related searches for query '{original_search_query}' from {website}")
        return {"success": True, "inserted_count": inserted_count}
    
    except Exception as e:
        logging.error(f"Failed to store related searches: {e}")
        if conn:
            conn.rollback()
        return {"success": False, "error": str(e)}
        
    finally:
        if conn:
            conn.close()

def get_related_searches(search_query: str = None, website: str = None, limit: int = 100):
    """
    Retrieve related searches from the database.
    
    Args:
        search_query: Filter by original search query (optional)
        website: Filter by website (optional)
        limit: Maximum number of results to return
        
    Returns:
        DataFrame with related searches
    """
    query = "SELECT * FROM related_searches WHERE 1=1"
    params = []
    
    if search_query:
        query += " AND original_search_query = %s"
        params.append(search_query)
        
    if website:
        query += " AND website = %s"
        params.append(website)
        
    query += " ORDER BY search_count DESC, inserted_at DESC LIMIT %s"
    params.append(limit)
    
    conn = None
    try:
        conn = get_db_connection()
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        logging.error(f"Error retrieving related searches: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close() 

def search_listings_by_similarity(search_term: str, search_fields: dict = None, limit: int = 100) -> pd.DataFrame:
    """
    Search listings in PostgreSQL using similarity on lowercase_title and optional filters.
    
    Args:
        search_term: Text to search for (will be matched against lowercase_title)
        search_fields: Dictionary of field filters to apply, with fields like:
            - date_range: date range tuple (start_date, end_date) or None
            - search_query: search query string or None
            - category: category string or None
            - usage: usage string or None
            - price_range: tuple of (min_price, max_price) or None
            - website: website string or None
            - company: company string or None
        limit: Maximum number of results to return
        
    Returns:
        DataFrame with matching listings
    """
    conn = None
    try:
        conn = get_db_connection()
        
        # Make sure pg_trgm extension is installed
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
            if not cur.fetchone():
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                conn.commit()
        
        # Start building the query with essential columns from both tables
        query = """
        SELECT 
            r.id as raw_id,
            p.id as preprocessed_id,
            r.title,
            r.description,
            r.search_query,
            r.category,
            r.usage,
            r.price,
            p.cleaned_price,
            r.website,
            r.company,
            p.parsed_date,
            r.link
        FROM 
            preprocessed_listings p
        JOIN 
            raw_listings r ON p.raw_listing_id = r.id
        WHERE 1=1
        """
        
        params = []
        
        # Apply similarity search on lowercase_title if search term is provided
        if search_term:
            # Convert search term to lowercase for case-insensitive matching
            search_term = search_term.lower()
            query += " AND similarity(p.lowercase_title, %s) > 0.3"
            params.append(search_term)
            # Add ordering by similarity
            query += " ORDER BY similarity(p.lowercase_title, %s) DESC"
            params.append(search_term)
        else:
            # Default ordering by parsed_date
            query += " ORDER BY p.parsed_date DESC"
        
        # Apply filters if provided
        if search_fields:
            # Date range filter
            if search_fields.get('date_range'):
                start_date, end_date = search_fields['date_range']
                if start_date:
                    query += " AND p.parsed_date >= %s"
                    params.append(start_date)
                if end_date:
                    query += " AND p.parsed_date <= %s"
                    params.append(end_date)
            
            # Search query filter
            if search_fields.get('search_query'):
                query += " AND r.search_query = %s"
                params.append(search_fields['search_query'])
            
            # Category filter
            if search_fields.get('category'):
                query += " AND r.category = %s"
                params.append(search_fields['category'])
            
            # Usage filter
            if search_fields.get('usage'):
                query += " AND r.usage = %s"
                params.append(search_fields['usage'])
            
            # Price range filter
            if search_fields.get('price_range'):
                min_price, max_price = search_fields['price_range']
                if min_price is not None:
                    query += " AND p.cleaned_price >= %s"
                    params.append(min_price)
                if max_price is not None:
                    query += " AND p.cleaned_price <= %s"
                    params.append(max_price)
            
            # Website filter
            if search_fields.get('website'):
                query += " AND r.website = %s"
                params.append(search_fields['website'])
            
            # Company filter
            if search_fields.get('company'):
                query += " AND r.company ILIKE %s"
                params.append(f"%{search_fields['company']}%")
        
        # Apply limit
        query += f" LIMIT {limit}"
        
        # Execute query
        with conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
            
            # Get column names
            column_names = [desc[0] for desc in cur.description]
            
            # Create DataFrame
            df = pd.DataFrame(results, columns=column_names)
            return df
    
    except Exception as e:
        logging.error(f"Error searching listings by similarity: {e}")
        return pd.DataFrame()
    
    finally:
        if conn:
            conn.close() 