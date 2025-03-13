import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from supabase import create_client, Client
import uuid

# Supabase connection details
SUPABASE_URL = "https://skyehzreswvdtbfuhupa.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNreWVoenJlc3d2ZHRiZnVodXBhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAzOTAwNjMsImV4cCI6MjA1NTk2NjA2M30.N9Hb72TxomkeFCXy4wZPXElSvKB2l5SHbHFN6yJgyFY"

def get_supabase_client() -> Client:
    """
    Creates and returns a Supabase client using connection details.
    
    Returns:
        Client: Supabase client
    """
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("Successfully created Supabase client")
        return supabase_client
    except Exception as e:
        logging.error(f"Error creating Supabase client: {e}")
        raise

def save_listing_to_supabase(
    title: str,
    price: str,
    description: str,
    link: str,
    date_col: str,
    usage: str,
    search_query: str,
    category: str,
    source_file: str,
    website: str,
    company: str
) -> Dict[str, Any]:
    """
    Save a listing to the Supabase 'raw_listings' table.
    
    Parameters:
    - title: Title of the listing
    - price: Price of the item
    - description: Description of the listing
    - link: URL to the listing
    - date_col: Date of the listing
    - usage: Usage condition (New, Used, etc.)
    - search_query: The search query used to find this listing
    - category: Category of the listing
    - source_file: Source file name
    - website: Website where the listing was found
    - company: Company/seller name
    
    Returns:
    - Response from Supabase
    """
    try:
        supabase_client = get_supabase_client()
        response = (
            supabase_client.from_("raw_listings")
            .insert(
                [
                    {
                        "uuid": str(uuid.uuid4()),
                        "title": title,
                        "price": price,
                        "description": description,
                        "link": link,
                        "date_col": date_col,
                        "usage": usage,
                        "search_query": search_query,
                        "category": category,
                        "source_file": source_file,
                        "website": website,
                        "company": company
                    }
                ]
            )
            .execute()
        )
        return {"success": True, "data": response}
    except Exception as e:
        logging.error(f"Error saving to Supabase: {e}")
        return {"success": False, "error": str(e)}

def save_multiple_listings_to_supabase(listings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Save multiple listings to the Supabase 'raw_listings' table.
    
    Parameters:
    - listings: A list of dictionaries, each containing the listing data
                Each dictionary should have keys matching the table columns
    
    Returns:
    - Response from Supabase
    """
    try:
        # Add UUID to each listing
        for listing in listings:
            if "uuid" not in listing:
                listing["uuid"] = str(uuid.uuid4())
        
        supabase_client = get_supabase_client()
        response = (
            supabase_client.from_("raw_listings")
            .insert(listings)
            .execute()
        )
        return {"success": True, "data": response}
    except Exception as e:
        logging.error(f"Error saving multiple listings to Supabase: {e}")
        return {"success": False, "error": str(e)}

def save_enriched_listing_to_supabase(
    title: str,
    price: str,
    description: str,
    link: str,
    date_col: str,
    usage: str,
    search_query: str,
    category: str,
    brand: str,
    model: str,
    classified_date: str,
    sentiment: str,
    sentiment_score: float,
    source_file: str,
    website: str = None,
    company: str = None
) -> Dict[str, Any]:
    """
    Save an enriched listing to the Supabase 'enriched_listings' table.
    
    Parameters:
    - All the enriched listing fields
    
    Returns:
    - Response from Supabase
    """
    try:
        supabase_client = get_supabase_client()
        response = (
            supabase_client.from_("enriched_listings")
            .insert(
                [
                    {
                        "uuid": str(uuid.uuid4()),
                        "title": title,
                        "price": price,
                        "description": description,
                        "link": link,
                        "date_col": date_col,
                        "usage": usage,
                        "search_query": search_query,
                        "category": category,
                        "brand": brand,
                        "model": model,
                        "classified_date": classified_date,
                        "sentiment": sentiment,
                        "sentiment_score": sentiment_score,
                        "source_file": source_file,
                        "website": website,
                        "company": company
                    }
                ]
            )
            .execute()
        )
        return {"success": True, "data": response}
    except Exception as e:
        logging.error(f"Error saving enriched listing to Supabase: {e}")
        return {"success": False, "error": str(e)}

def save_multiple_enriched_listings_to_supabase(listings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Save multiple enriched listings to the Supabase 'enriched_listings' table.
    
    Parameters:
    - listings: A list of dictionaries, each containing the enriched listing data
    
    Returns:
    - Response from Supabase
    """
    try:
        # Add UUID to each listing
        for listing in listings:
            if "uuid" not in listing:
                listing["uuid"] = str(uuid.uuid4())
        
        supabase_client = get_supabase_client()
        response = (
            supabase_client.from_("enriched_listings")
            .insert(listings)
            .execute()
        )
        return {"success": True, "data": response}
    except Exception as e:
        logging.error(f"Error saving multiple enriched listings to Supabase: {e}")
        return {"success": False, "error": str(e)}

def dataframe_to_supabase_raw(df: pd.DataFrame, source_file: str = None) -> Dict[str, Any]:
    """
    Convert a pandas DataFrame to the format expected by Supabase raw_listings table
    and save it to Supabase.
    
    Parameters:
    - df: DataFrame with the raw listing data
    - source_file: Source file name
    
    Returns:
    - Response from Supabase
    """
    if df.empty:
        logging.warning("DataFrame is empty. Nothing to store in Supabase.")
        return {"success": False, "error": "DataFrame is empty"}
        
    # Map DataFrame columns to Supabase columns
    listings = []
    
    for _, row in df.iterrows():
        listing = {
            "uuid": str(uuid.uuid4()),
            "title": row.get("Title", ""),
            "price": row.get("Price", ""),
            "description": row.get("Description", ""),
            "link": row.get("Link", ""),
            "date_col": row.get("Date", ""),
            "usage": row.get("Usage", ""),
            "search_query": row.get("search_query", None),
            "category": row.get("category", None),
            "source_file": source_file or row.get("source_file", None),
            "website": row.get("website", None),
            "company": row.get("company", None)
        }
        listings.append(listing)
        
    return save_multiple_listings_to_supabase(listings)

def dataframe_to_supabase_enriched(df: pd.DataFrame, source_file: str = None) -> Dict[str, Any]:
    """
    Convert a pandas DataFrame to the format expected by Supabase enriched_listings table
    and save it to Supabase.
    
    Parameters:
    - df: DataFrame with the enriched listing data
    - source_file: Source file name
    
    Returns:
    - Response from Supabase
    """
    if df.empty:
        logging.warning("DataFrame is empty. Nothing to store in Supabase.")
        return {"success": False, "error": "DataFrame is empty"}
        
    # Map DataFrame columns to Supabase columns
    listings = []
    
    for _, row in df.iterrows():
        listing = {
            "uuid": str(uuid.uuid4()),
            "title": row.get("Title", ""),
            "price": row.get("Price", ""),
            "description": row.get("Description", ""),
            "link": row.get("Link", ""),
            "date_col": row.get("Date", ""),
            "usage": row.get("Usage", ""),
            "search_query": row.get("search_query", None),
            "category": row.get("category", None),
            "brand": row.get("Brand", None),
            "model": row.get("Model", None),
            "classified_date": row.get("ClassifiedDate", None),
            "sentiment": row.get("Sentiment", None),
            "sentiment_score": row.get("SentimentScore", None),
            "source_file": source_file or row.get("source_file", None),
            "website": row.get("website", None),
            "company": row.get("company", None)
        }
        listings.append(listing)
        
    return save_multiple_enriched_listings_to_supabase(listings)

def search_listings_in_supabase(search_term: str, search_fields: dict = None, limit: int = 100) -> pd.DataFrame:
    """
    Search listings in Supabase with text match and optional filters.
    
    Args:
        search_term: Text to search for in the title
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
    try:
        supabase = get_supabase_client()
        if not supabase:
            return pd.DataFrame()
        
        # Start building the query
        query = supabase.from_("raw_listings").select("*")
        
        # Apply title search if provided
        if search_term:
            # Convert to lowercase for case-insensitive search
            search_term = search_term.lower()
            # Use ilike for case-insensitive partial matching
            query = query.ilike("title", f"%{search_term}%")
        
        # Apply filters if provided
        if search_fields:
            # Date range filter
            if search_fields.get('date_range'):
                start_date, end_date = search_fields['date_range']
                if start_date:
                    query = query.gte("date_col", start_date)
                if end_date:
                    query = query.lte("date_col", end_date)
            
            # Search query filter
            if search_fields.get('search_query'):
                query = query.eq("search_query", search_fields['search_query'])
            
            # Category filter
            if search_fields.get('category'):
                query = query.eq("category", search_fields['category'])
            
            # Usage filter
            if search_fields.get('usage'):
                query = query.eq("usage", search_fields['usage'])
            
            # Price range filter - note: this is approximate as we don't have cleaned_price in Supabase
            if search_fields.get('price_range'):
                min_price, max_price = search_fields['price_range']
                if min_price is not None:
                    # This is approximate and may not work well with string price formats
                    query = query.filter("price", "gte", str(min_price))
                if max_price is not None:
                    query = query.filter("price", "lte", str(max_price))
            
            # Website filter
            if search_fields.get('website'):
                query = query.eq("website", search_fields['website'])
            
            # Company filter
            if search_fields.get('company'):
                query = query.eq("company", search_fields['company'])
        
        # Apply limit and execute
        response = query.limit(limit).execute()
        
        # Convert to DataFrame
        if response.data:
            df = pd.DataFrame(response.data)
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error searching in Supabase: {e}")
        return pd.DataFrame() 