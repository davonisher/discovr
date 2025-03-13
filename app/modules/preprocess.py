import datetime
import re
from typing import Optional
import pandas as pd


def clean_price(price_str: str) -> float:
    if not isinstance(price_str, str):
        return 0.0

    # Remove all non-numeric and non-decimal characters
    cleaned = ''.join(char for char in price_str if char.isdigit() or char == ',' or char == '.')

    # Replace commas with periods for decimal separation
    cleaned = cleaned.replace(',', '.')

    # Handle cases like "Op aanvraag", "N.o.t.k.", etc.
    if cleaned.lower() in ["op aanvraag", "n.o.t.k.", "gereserveerd", "bieden"]:
        return 0.0

    # Split the cleaned string into parts
    parts = cleaned.split('.')

    # Handle cases like "€ 395,00" or "€35,00"
    if len(parts) == 2:
        try:
            return float('.'.join(parts))
        except ValueError:
            return 0.0

    # Handle cases like "€ 2.300,00"
    elif len(parts) == 1:
        try:
            return float(parts[0])
        except ValueError:
            return 0.0

    # If none of the cases match, return 0.0
    return 0.0


def parse_date(date_str: str) -> Optional[datetime.date]:
    """
    Parses a date string and returns a datetime.date object.
    
    Handles:
    - Relative dates: 'Vandaag' (Today), 'Gisteren' (Yesterday), 'Eergisteren' (The day before yesterday)
    - Absolute dates: e.g., '9 jan 25', '31 okt 24'
    
    Parameters:
    - date_str (str): The date string to parse.
    
    Returns:
    - datetime.date: The parsed date, or None if the date string is not recognized.
    """
    # Convert date_str to lowercase string
    date_str = str(date_str).strip().lower()
    today = datetime.date.today()

    # Handle relative dates
    relative_dates = {
        'vandaag': today,
        'gisteren': today - datetime.timedelta(days=1),
        'eergisteren': today - datetime.timedelta(days=2),
    }
    if date_str in relative_dates:
        return relative_dates[date_str]
    
    # Handle absolute dates with Dutch month abbreviations
    month_mapping = {
        'jan': 1, 'feb': 2, 'mrt': 3, 'apr': 4, 'mei': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'okt': 10, 'nov': 11, 'dec': 12
    }
    
    # Regex to match patterns like '9 jan 25'
    match = re.match(r'(\d{1,2})\s+([a-z]{3})\s+(\d{2})', date_str)
    if match:
        day, month_abbr, year_suffix = match.groups()
        month = month_mapping.get(month_abbr)
        if month:
            # Assuming year suffix '25' corresponds to 2025
            year = 2000 + int(year_suffix)
            try:
                return datetime.date(year, month, int(day))
            except ValueError:
                pass
    
    # If no match, return None
    return None


def preprocess_enriched_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the enriched DataFrame by converting date columns to datetime.
    
    Parameters:
    - df (pd.DataFrame): The enriched DataFrame.
    
    Returns:
    - pd.DataFrame: The preprocessed DataFrame.
    """
    if 'Date' in df.columns:
        df['Date'] = df['Date'].apply(parse_date)
    
    if 'Price' in df.columns:
        df['CleanedPrice'] = df['Price'].apply(clean_price)

   
    return df
