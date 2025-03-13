import asyncio
import os
import random
import pandas as pd
from playwright.async_api import async_playwright
#stealth async import
from playwright_stealth import stealth_async

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
        # Return a default user agent if file not found
        return ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"]

async def setup_browser_page(user_agent: str):
    """Setup and return configured browser page"""
    print(f"Setting up browser with user agent: {user_agent}")
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)  # Changed to headless=True
    context = await browser.new_context(
        user_agent=user_agent,
        viewport={'width': 1920, 'height': 1080}  # Added viewport
    )
    page = await context.new_page()
    await stealth_async(page)
    await page.route("**/*", block_aggressively)
    return p, browser, page

async def navigate_to_page(page, url: str, page_num: int):
    """Navigate to URL and handle errors"""
    try:
        await page.goto(url, timeout=60000)
        delay = random.uniform(6, 20)
        await asyncio.sleep(delay)  # Randomized delay
        await page.wait_for_load_state('networkidle')
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
        # Added wait before queries
        await asyncio.sleep(0.1)
        
        print("Extracting listing data...")
        title_el = await element.query_selector('.c-lot-card__title')
        price_el = await element.query_selector('.c-lot-card__price')
        desc_el = await element.query_selector('.c-lot-card__content')
        link_el = await element.query_selector('a.c-lot-card')
        status_el = await element.query_selector('.c-lot-card__status-text')
        timer_el = await element.query_selector('time.u-typography-h7')
        favorite_el = await element.query_selector('.FavoriteChip_content__kTemr')

        title_text = (await title_el.inner_text()).strip() if title_el else 'No title'
        price_text = (await price_el.inner_text()).strip() if price_el else 'No price'
        desc_text = (await desc_el.inner_text()).strip() if desc_el else 'No description'
        href = (await link_el.get_attribute('href')) if link_el else ''
        full_link = href if href else 'No link'
        status_text = (await status_el.inner_text()).strip() if status_el else 'No status'
        timer_text = (await timer_el.inner_text()).strip() if timer_el else 'No timer'
        favorites = (await favorite_el.inner_text()).strip() if favorite_el else '0'

        return {
            'Title': title_text,
            'Price': price_text,
            'Description': desc_text,
            'Link': full_link,
            'Status': status_text,
            'Time_Left': timer_text,
            'Favorites': favorites
        }
    except Exception as ex:
        print(f"Error extracting listing data: {ex}")
        return None

def build_url(base_url: str, search_query: str, page_num: int, category: str = None) -> str:
    """Build the URL for scraping based on search query, page number and optional category"""
    if category:
        return f"{base_url}/nl/c/{category}?q={search_query}&page={page_num}"
    return f"{base_url}/nl/s?q={search_query}&page={page_num}"

async def scrape_single_page(
    page_num: int,
    base_url: str,
    search_query: str,
    user_agent: str,
    listings: list,
    category: str = None
):
    """
    Scrape a single page of Catawiki asynchronously.
    Gathers title, price, description, link, and other details into 'listings' list.
    
    Args:
        page_num: Page number to scrape
        base_url: Base Catawiki URL
        search_query: Search term
        user_agent: User agent string
        listings: List to store results
        category: Optional category to filter results
    """
    url = build_url(base_url, search_query, page_num, category)
    
    try:
        p, browser, page = await setup_browser_page(user_agent)
        
        if not await navigate_to_page(page, url, page_num):
            print(f"Failed to navigate to page {page_num}, cleaning up...")
            await browser.close()
            await p.stop()
            return

        # Wait for listings to load
        listing_elements = await page.query_selector_all('article.c-lot-card__container')
        print(f"Found {len(listing_elements)} listings")
        
        for i, element in enumerate(listing_elements, 1):
            print(f"Processing listing {i}/{len(listing_elements)}")
            listing_data = await extract_listing_data(element)
            if listing_data:
                # Add category to listing data if provided
                if category:
                    listing_data['Category'] = category
                listings.append(listing_data)

        await browser.close()
        await p.stop()
    except Exception as e:
        print(f"Error in scrape_single_page: {e}")
        return

async def scrape_catawiki_playwright(search_query: str = "", max_pages: int = 5) -> pd.DataFrame:
    """
    Main async scraping function.
    Gathers data from up to 'max_pages' pages on Catawiki.
    Processes pages in smaller batches of 5 to prevent detection.
    """
    print(f"\nStarting scrape for search query: {search_query}")
    print(f"Will scrape up to {max_pages} pages")
    
    all_listings = []
    base_url = "https://www.catawiki.com"
    user_agents = load_user_agents()

    # Process pages in smaller batches of 5
    for batch_start in range(1, max_pages + 1, 5):
        batch_end = min(batch_start + 4, max_pages)
        
        tasks = []
        for page_num in range(batch_start, batch_end + 1):
            user_agent = random.choice(user_agents)
            print(f"Creating task for page {page_num}")
            task = scrape_single_page(
                page_num=page_num,
                base_url=base_url,
                search_query=search_query,
                user_agent=user_agent,
                listings=all_listings
            )
            tasks.append(task)

        print(f"Starting batch of {len(tasks)} tasks")
        await asyncio.gather(*tasks)
        
        # Longer delay between batches
        if batch_end < max_pages:
            delay = random.uniform(8, 12)
            print(f"Batch complete. Waiting {delay:.2f} seconds before next batch...")
            await asyncio.sleep(delay)

    print(f"\nScraping complete. Found {len(all_listings)} total listings")
    df = pd.DataFrame(all_listings)
    return df

if __name__ == "__main__":
    df = asyncio.run(scrape_catawiki_playwright())
    print(df)
    # Save results
    df.to_csv('catawiki_listings.csv', index=False)
