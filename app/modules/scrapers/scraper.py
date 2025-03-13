import asyncio
from playwright.async_api import async_playwright, TimeoutError
from playwright_stealth import stealth_async
import pandas as pd
import random
import urllib.parse
from app.utils import load_user_agents
import os
os.system("playwright install chromium")

# Function to block non-document resources
async def block_aggressively(route):
    if route.request.resource_type != "document":
        await route.abort()
    else:
        await route.continue_()

# User agents for rotating headers
USER_AGENTS = load_user_agents()
async def scrape_marktplaats_playwright(
    search_query: str,
    category: str,
    max_pages: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scrapes Marktplaats listings and related searches for given search query and category.
    
    Args:
        search_query: Search term to look for
        category: Category to search in
        max_pages: Maximum number of pages to scrape (default 1)
        
    Returns:
        Tuple containing:
        - DataFrame with listings
        - DataFrame with related searches
    """
    listings = []
    related_searches = []
    base_url = "https://www.marktplaats.nl"
    search_query_encoded = urllib.parse.quote(search_query)

    for page_num in range(1, max_pages + 1):
        url = f"{base_url}/l/{category}/q/{search_query_encoded}/p/{page_num}/"
        
        async with async_playwright() as p:
            user_agent = random.choice(USER_AGENTS)
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=user_agent)
            page = await context.new_page()
            await stealth_async(page)
            await page.route("**/*", block_aggressively)

            try:
                await page.goto(url, timeout=60000)
                await asyncio.sleep(2)
                await page.wait_for_load_state('networkidle')

                # Get related searches
                related_elements = await page.query_selector_all('div.hz-SuggestedSearches-content a')
                for related in related_elements:
                    title = await related.inner_text()
                    href = await related.get_attribute('href')
                    
                    # Extract category and subcategory
                    category_href = 'No category'
                    subcategory_href = 'No subcategory'
                    if href and '/l/' in href:
                        parts = href.split('/l/')[1].split('/')
                        if len(parts) >= 2:
                            category_href = parts[0]
                            if '/' in parts[1]:
                                subcategory_href = parts[1].split('/')[0]

                    related_searches.append({
                        'Title': title,
                        'Link': href,
                        'Category': category_href,
                        'Subcategory': subcategory_href,
                        'Search_Query': search_query,
                        'Original_Category': category
                    })

            except TimeoutError:
                print(f"Timeout occurred while loading {url}")
                continue
            except Exception as e:
                print(f"Error occurred while loading {url}: {e}")
                continue

            listing_elements = await page.query_selector_all('li[class*="Listing"]')

            for element in listing_elements:
                try:
                    # Extract listing details
                    title = await element.query_selector('h3[class*="Listing-title"]')
                    title_text = await title.inner_text() if title else 'No title'
                    title_text = title_text.strip()

                    price = await element.query_selector('p[class*="Listing-price"]')
                    price_text = await price.inner_text() if price else 'No price'
                    price_text = price_text.strip()

                    usage = await element.query_selector('span[class*="hz-Attribute hz-Attribute--default"]')
                    usage_text = await usage.inner_text() if usage else 'No usage'
                    usage_text = usage_text.strip()

                    description = await element.query_selector('p[class*="Listing-description"]')
                    description_text = await description.inner_text() if description else 'No description'
                    description_text = description_text.strip()

                    link = await element.query_selector('a[class*="Link"]')
                    href = await link.get_attribute('href') if link else None
                    full_link = base_url + href if href and href.startswith('/v/') else 'No link'

                    date = await element.query_selector('span[class*="Listing-date"]')
                    date_text = await date.inner_text() if date else 'No date'
                    date_text = date_text.strip()

                    website_link = await element.query_selector('a[class*="hz-Link hz-Link--isolated hz-Listing-sellerCoverLink"]')
                    website_url = await website_link.get_attribute('href') if website_link else None
                    is_company = 'Yes' if website_url else 'No'

                    location = await element.query_selector('span[class*="hz-Listing-distance-label"]')
                    location_text = await location.inner_text() if location else 'No location'
                    location_text = location_text.strip()

                    seller_name = await element.query_selector('a.hz-Link.hz-Link--isolated.hz-TextLink span.hz-Listing-seller-name')
                    seller_text = await seller_name.inner_text() if seller_name else 'No seller'
                    seller_text = seller_text.strip()

                    # Extract category and subcategory from href
                    link_element = await element.query_selector('a[class*="Link"]')
                    href = await link_element.get_attribute('href') if link_element else None
                    category_href = 'No category'
                    subcategory_href = 'No subcategory'
                    if href and href.startswith('/v/'):
                        parts = href.split('/v/')[1].split('/')
                        if len(parts) >= 2:
                            category_href = parts[0]
                            subcategory_href = parts[1]

                    listings.append({
                        'Title': title_text,
                        'Price': price_text,
                        'Description': description_text,
                        'Link': full_link,
                        'Date': date_text,
                        'Usage': usage_text,
                        'Website': website_url if website_url else 'No website',
                        'Company': is_company,
                        'Location': location_text,
                        'Seller': seller_text,
                        'Category': category_href,
                        'Subcategory': subcategory_href,
                        'search_query': search_query,
                        'category': category
                    })

                except Exception as e:
                    print(f"Error parsing listing: {e}")
                    continue

            await browser.close()
    
    df = pd.DataFrame(listings)
    related_df = pd.DataFrame(related_searches)
    
    # Save related searches
    related_df.to_csv(f'/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Professioneel/Coding projects/marktplaats/data/related_searches/related_search_{search_query}_{category}.csv', index=False)
    
    return df, related_df
