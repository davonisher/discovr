import logging
import pandas as pd
from datetime import datetime, timedelta
import re
from typing import Optional, Dict, Any, List, Tuple

# Use relative imports that will work both when running from app directory and project root
try:
    # Try importing with app prefix (for running from project root)
    from app.modules.db.ps.database import (
        get_db_connection,
        check_file_processed,
        store_preprocessed_data
    )
except ImportError:
    # Fall back to direct import (for running from app directory)
    from modules.db.ps.database import (
        get_db_connection,
        check_file_processed,
        store_preprocessed_data
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_price(price_str: str) -> float:
    """
    Convert a price string to a float value.
    
    Args:
        price_str: Price as a string (e.g., "€ 395,00")
        
    Returns:
        Cleaned price as a float
    """
    if not isinstance(price_str, str):
        return 0.0

    # Remove all non-numeric and non-decimal characters
    cleaned = ''.join(char for char in price_str if char.isdigit() or char == ',' or char == '.')

    # Handle empty string case
    if not cleaned:
        return 0.0
        
    # Handle European number format with comma as decimal separator
    if ',' in cleaned and '.' in cleaned:
        # For formats like 1.215,00 - remove the dots first, then replace comma with dot
        cleaned = cleaned.replace('.', '')
        cleaned = cleaned.replace(',', '.')
    elif ',' in cleaned:
        # For formats like 215,00 - replace comma with dot
        cleaned = cleaned.replace(',', '.')
        
    # Try to convert to float
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date strings in various formats to datetime.
    
    Handles:
    - Standard date formats (01-01-2023, 2023-01-01, etc.)
    - Dutch date formats with month names (01 jan 2023, 1 januari 2023)
    - Relative dates: 'Vandaag' (Today), 'Gisteren' (Yesterday), 'Eergisteren' (Day before yesterday)
    - Relative expressions: '2 dagen geleden' (2 days ago), '1 week geleden' (1 week ago)
    - Short year formats: '9 jan 25' (January 9, 2025)
    
    Args:
        date_str: Date as a string
        
    Returns:
        Parsed datetime or None if parsing failed
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    # Clean and normalize the input
    date_str = str(date_str).strip().lower()
    
    # Handle common invalid values
    if date_str in ['nan', 'none', 'null', 'no date', '']:
        return None
    
    # Current date reference
    today = datetime.now()
    
    # 1. Handle relative dates
    relative_dates = {
        'vandaag': today,
        'today': today,
        'gisteren': today - timedelta(days=1),
        'yesterday': today - timedelta(days=1),
        'eergisteren': today - timedelta(days=2),
    }
    
    if date_str in relative_dates:
        return relative_dates[date_str]
    
    # 2. Handle relative expressions (X days/weeks ago)
    
    # "X dagen geleden" / "X days ago"
    days_ago_match = re.search(r'(\d+)\s*(dag(en)?|days?)\s*(geleden|ago)', date_str)
    if days_ago_match:
        days = int(days_ago_match.group(1))
        return today - timedelta(days=days)
    
    # "X weken geleden" / "X weeks ago"
    weeks_ago_match = re.search(r'(\d+)\s*(we(e)?k(en)?|weeks?)\s*(geleden|ago)', date_str)
    if weeks_ago_match:
        weeks = int(weeks_ago_match.group(1))
        return today - timedelta(weeks=weeks)
    
    # 3. Handle short Dutch date format like '9 jan 25'
    month_mapping = {
        'jan': 1, 'feb': 2, 'mrt': 3, 'mar': 3, 'apr': 4, 'mei': 5, 'may': 5, 
        'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'okt': 10, 'oct': 10, 
        'nov': 11, 'dec': 12
    }
    
    # 3.1 First try the short format pattern like '9 jan 25'
    short_date_match = re.match(r'(\d{1,2})\s+([a-z]{3,4})\s+(\d{2})', date_str)
    if short_date_match:
        day, month_abbr, year_suffix = short_date_match.groups()
        month = month_mapping.get(month_abbr[:3])  # Take first 3 chars of month
        if month:
            # Assuming year suffix '25' corresponds to 2025
            year = 2000 + int(year_suffix)
            try:
                return datetime(year, month, int(day))
            except ValueError:
                pass  # Continue to other formats if this fails
    
    # 4. Handle date formats with Dutch month names
    
    # Map Dutch month names to English for parsing
    dutch_to_english = {
        # Short forms
        'jan': 'Jan', 'feb': 'Feb', 'mrt': 'Mar', 'apr': 'Apr',
        'mei': 'May', 'jun': 'Jun', 'jul': 'Jul', 'aug': 'Aug',
        'sep': 'Sep', 'okt': 'Oct', 'nov': 'Nov', 'dec': 'Dec',
        
        # Full forms
        'januari': 'January', 'februari': 'February', 'maart': 'March',
        'april': 'April', 'mei': 'May', 'juni': 'June',
        'juli': 'July', 'augustus': 'August', 'september': 'September',
        'oktober': 'October', 'november': 'November', 'december': 'December'
    }
    
    # Convert Dutch month names to English
    normalized_date_str = date_str
    
    # 4.1 Special handling for full Dutch month names with a space-separated format
    # Match patterns like '15 augustus 2023' or '1 januari 2024'
    full_month_match = re.match(r'(\d{1,2})\s+([a-z]+)\s+(\d{4})', date_str)
    if full_month_match:
        day, month_name, year = full_month_match.groups()
        if month_name in dutch_to_english:
            english_month = dutch_to_english[month_name]
            try:
                month_num = datetime.strptime(english_month, '%B').month
                return datetime(int(year), month_num, int(day))
            except ValueError:
                try:
                    month_num = datetime.strptime(english_month, '%b').month
                    return datetime(int(year), month_num, int(day))
                except ValueError:
                    pass  # Continue to other parsing methods
    
    # 4.2 Try to replace Dutch month names with English equivalents for other formats
    for dutch, english in dutch_to_english.items():
        if dutch in normalized_date_str:
            normalized_date_str = normalized_date_str.replace(dutch, english)
    
    # 5. Try common date formats
    date_formats = [
        # Dutch/European formats
        "%d-%m-%Y",  # 01-01-2023
        "%d/%m/%Y",  # 01/01/2023
        "%d.%m.%Y",  # 01.01.2023
        "%d %m %Y",  # 01 01 2023
        
        # With month names
        "%d %b %Y",  # 01 Jan 2023
        "%d %B %Y",  # 01 January 2023
        "%d-%b-%Y",  # 01-Jan-2023
        "%d-%B-%Y",  # 01-January-2023
        
        # International formats
        "%Y-%m-%d",  # 2023-01-01 (ISO)
        "%Y/%m/%d",  # 2023/01/01
        "%m/%d/%Y",  # 01/01/2023 (US)
        "%b %d, %Y",  # Jan 01, 2023
        "%B %d, %Y",  # January 01, 2023
        
        # Two-digit year formats
        "%d-%m-%y",  # 01-01-23
        "%d/%m/%y",  # 01/01/23
        "%y-%m-%d",  # 23-01-01
        "%y/%m/%d",  # 23/01/01
        
        # Additional formats with day and month
        "%d %B",      # 1 January (assumed current year)
        "%d %b",      # 1 Jan (assumed current year)
        "%d-%m",      # 01-01 (assumed current year)
        "%d/%m",      # 01/01 (assumed current year)
    ]
    
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(normalized_date_str, fmt)
            
            # For formats without year, set the year to current year
            if "%Y" not in fmt and "%y" not in fmt:
                parsed_date = parsed_date.replace(year=today.year)
                
                # If the resulting date is in the future by more than a month, 
                # assume it's from the previous year
                if (parsed_date - today).days > 31:
                    parsed_date = parsed_date.replace(year=today.year - 1)
                    
            return parsed_date
        except ValueError:
            continue
    
    # 6. Last resort: Try to find any date-like pattern in the string
    # This helps with unusual formats or dates with surrounding text
    
    # Look for day/month/year patterns
    date_pattern = re.search(r'(\d{1,2})[-/. ](\d{1,2}|[a-z]{3,})[-/. ](\d{2,4})', date_str)
    if date_pattern:
        day, month, year = date_pattern.groups()
        
        # Try to parse the month if it's a string
        if not month.isdigit():
            month = month[:3].lower()  # Take first 3 chars of month name
            month_num = month_mapping.get(month)
            if not month_num:
                return None
        else:
            month_num = int(month)
            
        # Handle 2-digit years
        if len(year) == 2:
            year = 2000 + int(year)
        
        try:
            return datetime(int(year), month_num, int(day))
        except ValueError:
            pass  # Invalid date
    
    # If all parsing attempts fail
    logger.debug(f"Failed to parse date string: {date_str}")
    return None

def process_raw_listing(listing_id: int, title: str, price: str, description: str, 
                       link: str, date_col: str, usage: str, search_query: str,
                       category: str, source_file: str, website: str, company: str) -> Dict[str, Any]:
    """
    Process a single raw listing and return preprocessed data.
    
    Args:
        listing_id: ID of the raw listing
        Various fields from the raw listing
        
    Returns:
        Dictionary with preprocessed data (only the processed fields, not duplicating raw data)
    """
    # Clean price
    cleaned_price = clean_price(price)
    
    # Parse date
    parsed_date = parse_date(date_col)
    
    # Lowercase title and description for case-insensitive processing
    lowercase_title = title.lower() if title else None
    lowercase_description = description.lower() if description else None
    
    # Return preprocessed data - only the fields we need to store
    return {
        "raw_listing_id": listing_id,
        "cleaned_price": cleaned_price,
        "parsed_date": parsed_date,
        "lowercase_title": lowercase_title,
        "lowercase_description": lowercase_description,
        "processing_status": "processed"
        # We don't need to include the other fields from raw_listings since we have the raw_listing_id
    }

def process_raw_listings_batch(batch_size: int = 100, limit: int = 500, filter_clause: str = None) -> Tuple[int, int]:
    """
    Process a batch of raw listings and store in preprocessed_listings table.
    
    Args:
        batch_size: Number of raw listings to process in each batch
        limit: Maximum number of listings to process in total
        filter_clause: Optional SQL WHERE clause to filter raw listings (without the 'WHERE' keyword)
        
    Returns:
        Tuple of (processed_count, error_count)
    """
    conn = None
    processed_count = 0
    error_count = 0
    
    try:
        conn = get_db_connection()
        
        # Get raw listings that haven't been processed yet
        with conn.cursor() as cur:
            # Start with the base query
            query = """
                SELECT r.id, r.title, r.price, r.description, r.link, r.date_col, r.usage,
                       r.search_query, r.category, r.source_file, r.website, r.company
                FROM raw_listings r
                LEFT JOIN preprocessed_listings p ON r.id = p.raw_listing_id
                WHERE p.id IS NULL
            """
            
            # Add the filter clause if provided
            if filter_clause:
                # Use regex to properly qualify column references to avoid ambiguity
                # This handles variations in spacing around the = operator
                modified_filter = filter_clause
                
                # Replace search_query references
                modified_filter = re.sub(r'(\bsearch_query\b)(\s*[=<>])', r'r.\1\2', modified_filter)
                
                # Replace category references
                modified_filter = re.sub(r'(\bcategory\b)(\s*[=<>])', r'r.\1\2', modified_filter)
                
                # Replace title references
                modified_filter = re.sub(r'(\btitle\b)(\s*[=<>])', r'r.\1\2', modified_filter)
                
                # Replace LIKE operator cases
                modified_filter = re.sub(r'(\bsearch_query\b)(\s+LIKE)', r'r.\1\2', modified_filter, flags=re.IGNORECASE)
                modified_filter = re.sub(r'(\bcategory\b)(\s+LIKE)', r'r.\1\2', modified_filter, flags=re.IGNORECASE)
                modified_filter = re.sub(r'(\btitle\b)(\s+LIKE)', r'r.\1\2', modified_filter, flags=re.IGNORECASE)
                
                # Replace IN operator cases
                modified_filter = re.sub(r'(\bsearch_query\b)(\s+IN)', r'r.\1\2', modified_filter, flags=re.IGNORECASE)
                modified_filter = re.sub(r'(\bcategory\b)(\s+IN)', r'r.\1\2', modified_filter, flags=re.IGNORECASE)
                modified_filter = re.sub(r'(\btitle\b)(\s+IN)', r'r.\1\2', modified_filter, flags=re.IGNORECASE)
                
                logger.info(f"Original filter: {filter_clause}")
                logger.info(f"Modified filter: {modified_filter}")
                
                query += f" AND {modified_filter}"
                
            # Add the order and limit clauses
            query += " ORDER BY r.inserted_at DESC LIMIT %s"
            
            cur.execute(query, (limit,))
            
            raw_listings = cur.fetchall()
            
        if not raw_listings:
            logger.info("No new raw listings to process")
            return (0, 0)
            
        logger.info(f"Found {len(raw_listings)} raw listings to process")
        
        # Process listings in smaller batches to avoid losing everything on error
        current_batch = []
        
        for i, raw in enumerate(raw_listings):
            try:
                # Process individual raw listing
                preprocessed = process_raw_listing(
                    listing_id=raw[0],
                    title=raw[1],
                    price=raw[2],
                    description=raw[3],
                    link=raw[4],
                    date_col=raw[5],
                    usage=raw[6],
                    search_query=raw[7],
                    category=raw[8],
                    source_file=raw[9],
                    website=raw[10],
                    company=raw[11]
                )
                
                current_batch.append(preprocessed)
                
                # Process batch if batch size reached or last item
                if len(current_batch) >= batch_size or i == len(raw_listings) - 1:
                    # Only try to process the batch if it's not empty
                    if current_batch:
                        try:
                            # Insert batch into preprocessed_listings with a new connection
                            batch_conn = None
                            try:
                                batch_conn = get_db_connection()
                                with batch_conn.cursor() as cur:
                                    # Prepare each item for insert with proper error handling
                                    valid_items = []
                                    for item in current_batch:
                                        try:
                                            valid_items.append(cur.mogrify("(%s,%s,%s,%s,%s,%s,%s)", (
                                                item["raw_listing_id"],
                                                item["cleaned_price"],
                                                item["parsed_date"],
                                                'processed',  # processing_status
                                                datetime.now(),  # inserted_at
                                                None,  # Additional data field 1 (placeholder)
                                                None   # Additional data field 2 (placeholder)
                                            )).decode('utf-8'))
                                        except Exception as item_error:
                                            logger.error(f"Error preparing item for insert: {item_error}")
                                            error_count += 1
                                    
                                    # Only proceed if we have valid items
                                    if valid_items:
                                        args_str = ','.join(valid_items)
                                        
                                        cur.execute(f"""
                                            INSERT INTO preprocessed_listings 
                                            (raw_listing_id, cleaned_price, parsed_date, processing_status, inserted_at, additional_data1, additional_data2)
                                            VALUES {args_str}
                                            ON CONFLICT (raw_listing_id) DO UPDATE 
                                            SET processing_status = 'processed',
                                                cleaned_price = EXCLUDED.cleaned_price,
                                                parsed_date = EXCLUDED.parsed_date
                                        """)
                                        
                                        batch_conn.commit()
                                        processed_count += len(valid_items)
                                        logger.info(f"Inserted {len(valid_items)} preprocessed listings")
                                    else:
                                        logger.warning("No valid items to insert in this batch")
                            except Exception as batch_e:
                                logger.error(f"Error inserting batch: {batch_e}")
                                error_count += len(current_batch)
                                if batch_conn:
                                    batch_conn.rollback()
                            finally:
                                if batch_conn:
                                    batch_conn.close()
                        except Exception as batch_e:
                            logger.error(f"Error processing batch: {batch_e}")
                            error_count += len(current_batch)
                    
                    # Clear the batch after processing
                    current_batch = []
                
            except Exception as e:
                logger.error(f"Error processing raw listing {raw[0]}: {e}")
                error_count += 1
                
        logger.info(f"Processed {processed_count} raw listings with {error_count} errors")
        
    except Exception as e:
        logger.error(f"Error in process_raw_listings_batch: {e}")
    finally:
        if conn:
            conn.close()
            
    return (processed_count, error_count)

if __name__ == "__main__":
    # This allows running the preprocessing as a standalone script
    processed, errors = process_raw_listings_batch(batch_size=100, limit=1000)
    print(f"Processed {processed} raw listings with {errors} errors") 