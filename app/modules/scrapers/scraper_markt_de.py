import asyncio
import os
import random
import pandas as pd
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
import urllib.parse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f"Project root directory: {project_root}")

USER_AGENTS_PATH = os.path.join(project_root, "data", "user_agents.txt")
print(f"User agents file path: {USER_AGENTS_PATH}")

#########################
#  SCRAPING FUNCTIONS   #
#########################
async def block_aggressively(route):
    """
    Prevent the browser from downloading images, CSS, etc.
    This speeds up scraping if you only need the HTML.
    """
    if route.request.resource_type != "document":
        await route.abort()
    else:
        await route.continue_()

def load_user_agents() -> list:
    """
    Load user agents from a text file, one per line.
    """
    try:
        with open(USER_AGENTS_PATH, "r") as f:
            agents = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(agents)} user agents")
            return agents
    except FileNotFoundError:
        print("Warning: User agents file not found, using default agent")
        return ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"]

async def setup_browser_page(user_agent: str):
    """Setup and return configured browser page"""
    print(f"Setting up browser with user agent: {user_agent}")
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        user_agent=user_agent,
        viewport={'width': 1920, 'height': 1080}
    )
    page = await context.new_page()
    print("Applying stealth mode")
    await stealth_async(page)
    await page.route("**/*", block_aggressively)
    return p, browser, page

async def navigate_to_page(page, url: str, page_num: int):
    """Navigate to URL and handle errors"""
    print(f"Navigating to page {page_num}: {url}")
    try:
        await page.goto(url, timeout=60000)
        delay = random.uniform(6, 20)
        print(f"Waiting {delay:.2f} seconds...")
        await asyncio.sleep(delay)
        await page.wait_for_load_state('networkidle')
        print(f"Successfully loaded page {page_num}")
        return True
    except TimeoutError:
        print(f"Timeout at page {page_num}, url={url}")
        return False
    except Exception as e:
        print(f"Error loading page {page_num}: {e}")
        return False

async def extract_listing_data(element):
    """Extract data from a single listing element"""
    try:
        await asyncio.sleep(0.1)
        
        print("Extracting listing data...")
        title_el = await element.query_selector('.clsy-c-result-list-item__title')
        price_el = await element.query_selector('.clsy-c-result-list-item__price-amount')
        desc_el = await element.query_selector('.clsy-c-result-list-item__description')
        link_el = await element.query_selector('.clsy-c-result-list-item__link')
        date_el = await element.query_selector('.clsy-c-result-list-item__date')
        location_el = await element.query_selector('.clsy-c-result-list-item__location')
        category_el = await element.query_selector('.clsy-c-result-list-item__category')

        title_text = (await title_el.inner_text()).strip() if title_el else 'No title'
        price_text = (await price_el.inner_text()).strip() if price_el else 'No price'
        desc_text = (await desc_el.inner_text()).strip() if desc_el else 'No description'
        href = (await link_el.get_attribute('href')) if link_el else ''
        full_link = href if href.startswith('http') else f"https://www.markt.de{href}" if href else 'No link'
        date_text = (await date_el.inner_text()).strip() if date_el else 'No date'
        location_text = (await location_el.inner_text()).strip() if location_el else 'No location'
        category_text = (await category_el.inner_text()).strip() if category_el else 'No category'

        print(f"Extracted listing: {title_text[:50]}...")
        return {
            'Title': title_text,
            'Price': price_text,
            'Description': desc_text,
            'Link': full_link,
            'Date': date_text,
            'Location': location_text,
            'Category': category_text
        }
    except Exception as ex:
        print(f"Error extracting listing data: {ex}")
        return None

async def scrape_single_page(
    page_num: int,
    base_url: str,
    search_query: str,
    user_agent: str, 
    listings: list
):
    """
    Scrape a single page of Markt.de asynchronously.
    """
    url = f"{base_url}/k/{search_query}/?page={page_num}"
    print(f"\nScraping page {page_num}")
    
    try:
        p, browser, page = await setup_browser_page(user_agent)
        
        if not await navigate_to_page(page, url, page_num):
            print(f"Failed to navigate to page {page_num}, cleaning up...")
            await browser.close()
            await p.stop()
            return

        print("Looking for listing elements...")
        listing_elements = await page.query_selector_all('li.clsy-c-result-list-item')
        print(f"Found {len(listing_elements)} listings")
        
        for i, element in enumerate(listing_elements, 1):
            print(f"Processing listing {i}/{len(listing_elements)}")
            listing_data = await extract_listing_data(element)
            if listing_data:
                listings.append(listing_data)

        print(f"Completed page {page_num}, cleaning up...")
        await browser.close()
        await p.stop()
    except Exception as e:
        print(f"Error in scrape_single_page: {e}")
        return

async def scrape_markt_playwright(search_query: str, max_pages: int = 5) -> pd.DataFrame:
    """
    Main async scraping function for Markt.de.
    """
    print(f"\nStarting scrape for search query: {search_query}")
    print(f"Will scrape up to {max_pages} pages")
    
    all_listings = []
    base_url = "https://www.markt.de"
    user_agents = load_user_agents()
    search_query_encoded = urllib.parse.quote(search_query)

    for batch_start in range(1, max_pages + 1, 5):
        batch_end = min(batch_start + 4, max_pages)
        print(f"\nProcessing batch: pages {batch_start} to {batch_end}")
        
        tasks = []
        for page_num in range(batch_start, batch_end + 1):
            user_agent = random.choice(user_agents)
            print(f"Creating task for page {page_num}")
            task = scrape_single_page(
                page_num=page_num,
                base_url=base_url,
                search_query=search_query_encoded,
                user_agent=user_agent,
                listings=all_listings
            )
            tasks.append(task)

        print(f"Starting batch of {len(tasks)} tasks")
        await asyncio.gather(*tasks)
        
        if batch_end < max_pages:
            delay = random.uniform(8, 12)
            print(f"Batch complete. Waiting {delay:.2f} seconds before next batch...")
            await asyncio.sleep(delay)

    print(f"\nScraping complete. Found {len(all_listings)} total listings")
    df = pd.DataFrame(all_listings)
    return df

if __name__ == "__main__":
    print("Starting Markt.de scraper...")
    df = asyncio.run(scrape_markt_playwright(search_query="volkswagen+golf"))
    print("\nFinal results:")
    print(df)
    print("Saving results to CSV...")
    df.to_csv('markt_listings.csv', index=False)
