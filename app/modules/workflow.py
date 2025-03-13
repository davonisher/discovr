import asyncio
import pandas as pd
from modules.scrapers.scraper import scrape_marktplaats_playwright
from modules.structured_classifier import classify_listing_with_llm

async def do_scrape_and_classify(search_query: str, category: str, max_pages: int = 1) -> pd.DataFrame:
    """
    1. Scrape listings via scrape_marktplaats_playwright.
    2. Als 'Usage' == 'gebruikt', roep classify_listing_with_llm aan
       om merk, model en datum te verrijken.
    3. Retourneer verrijkt DataFrame.
    """
    # 1. Scrape
    df_listings = await scrape_marktplaats_playwright(
        search_query=search_query,
        category=category,
        max_pages=max_pages
    )

    if df_listings.empty:
        return df_listings  # Return empty if no results

    # 2. Build a list of enriched rows
    results = []
    for _, row in df_listings.iterrows():
        usage_str = str(row.get("Usage", "")).lower()
        title     = row.get("Title", "")
        description = row.get("Description", "")
        date_str  = row.get("Date", "")

        # Copy existing fields
        row_dict = {
            "Title": title,
            "Price": row.get("Price", ""),
            "Description": description,
            "Link": row.get("Link", ""),
            "Date": date_str,
            "Usage": row.get("Usage", ""),
            "Brand": None,
            "Model": None,
            "ClassifiedDate": None,
        }

        # Only classify if 'Usage' == 'gebruikt'
        if usage_str == "gebruikt":
            print(f"Calling classifier for {title}")
            llm_dict = await classify_listing_with_llm(title, description, date_str)
            if llm_dict:
                row_dict["Brand"] = llm_dict.get("brand", "Unknown")
                row_dict["Model"] = llm_dict.get("model", "Unknown")
                row_dict["ClassifiedDate"] = llm_dict.get("date", date_str)
        else:
            row_dict["Brand"] = None
            row_dict["Model"] = None
            row_dict["ClassifiedDate"] = None

        results.append(row_dict)

    # 3. Convert results to a new DataFrame
    df2 = pd.DataFrame(results)
    return df2
