import asyncio
import os
import random
import pandas as pd
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

USER_AGENTS_PATH = "/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Professioneel/Coding projects/marktplaats/app/data/user-agents.txt"

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
            return agents
    except FileNotFoundError:
        print("Warning: User agents file not found, using default agent")
        return ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"]

async def setup_browser_page(user_agent: str):
    """Setup and return configured browser page"""
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        user_agent=user_agent,
        viewport={'width': 1920, 'height': 1080}
    )
    page = await context.new_page()
    await stealth_async(page)
    return p, browser, page

async def navigate_to_page(page, url: str, page_num: int):
    """Navigate to URL and handle errors"""
    try:
        await page.goto(url, timeout=60000)
        delay = random.uniform(3, 7)
        await asyncio.sleep(delay)
        
        for i in range(3):
            await page.evaluate("window.scrollBy(0, 300)")
            scroll_delay = random.uniform(1, 3)
            await asyncio.sleep(scroll_delay)
            
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
        title_el = await element.query_selector('.s-item__title')
        price_el = await element.query_selector('.s-item__price')
        condition_el = await element.query_selector('.s-item__subtitle')
        link_el = await element.query_selector('.s-item__link')
        location_el = await element.query_selector('.s-item__location')
        shipping_el = await element.query_selector('.s-item__shipping')

        title = await title_el.inner_text() if title_el else 'No title'
        price = await price_el.inner_text() if price_el else 'No price'
        condition = await condition_el.inner_text() if condition_el else 'No condition'
        link = await link_el.get_attribute('href') if link_el else 'No link'
        location = await location_el.inner_text() if location_el else 'No location'
        shipping = await shipping_el.inner_text() if shipping_el else 'No shipping info'

        return {
            'Title': title,
            'Price': price,
            'Condition': condition,
            'Link': link,
            'Location': location,
            'Shipping': shipping
        }
    except Exception as ex:
        print(f"Error extracting listing data: {ex}")
        return None

async def scrape_single_page(
    page_num: int,
    base_url: str,
    search_query_encoded: str,
    user_agent: str,
    listings: list
):
    """
    Scrape a single page of eBay asynchronously.
    """
    url = f"{base_url}/sch/i.html?_nkw={search_query_encoded}&_sacat=0&_from=R40&_pgn={page_num}"
    
    try:
        p, browser, page = await setup_browser_page(user_agent)
        
        if not await navigate_to_page(page, url, page_num):
            await browser.close()
            await p.stop()
            return

        listing_elements = await page.query_selector_all('.s-item')
        
        for i, element in enumerate(listing_elements, 1):
            listing_data = await extract_listing_data(element)
            if listing_data:
                listings.append(listing_data)

        await browser.close()
        await p.stop()
    except Exception as e:
        print(f"Error in scrape_single_page: {e}")
        return

async def scrape_ebay_playwright(search_query: str = "", max_pages: int = 2) -> pd.DataFrame:
    """
    Main scraping function for eBay.
    """
    all_listings = []
    base_url = "https://www.ebay.nl"
    user_agents = load_user_agents()
    search_query_encoded = search_query.replace(" ", "+") if search_query else ""

    user_agent = random.choice(user_agents)
    await scrape_single_page(
        page_num=1,
        base_url=base_url,
        search_query_encoded=search_query_encoded,
        user_agent=user_agent,
        listings=all_listings
    )

    df = pd.DataFrame(all_listings)
    return df

if __name__ == "__main__":
    print("Starting eBay scraper...")
    df = asyncio.run(scrape_ebay_playwright(search_query="supreme"))
    print("\nFinal results:")
    print(df)
    print("Saving results to CSV...")
    df.to_csv('ebay_listings.csv', index=False)
    print("Done!")
