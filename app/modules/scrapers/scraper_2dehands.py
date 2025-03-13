import asyncio
from playwright.async_api import async_playwright, TimeoutError
from playwright_stealth import stealth_async
import pandas as pd
import random
import urllib.parse

# Functie om niet-document resources te blokkeren
async def block_aggressively(route):
    if route.request.resource_type != "document":
        await route.abort()
    else:
        await route.continue_()

# User agents voor roterende headers
with open("/data/user-agents.txt") as f:
    USER_AGENTS = f.read().splitlines()

async def scrape_2dehands_playwright(
    search_query,
    category,
    max_pages=1
):
    listings = []
    base_url = "https://www.2dehands.be"
    search_query_encoded = urllib.parse.quote(search_query)
    for page_num in range(1, max_pages + 1):
        url = f"{base_url}/l/{category}/q/{search_query_encoded}/p/{page_num}/"
        async with async_playwright() as p:
            user_agent = random.choice(USER_AGENTS)
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=user_agent)
            page = await context.new_page()
            await stealth_async(page)  # Pas stealth-modus to
            await page.route("**/*", block_aggressively)
            try:
                await page.goto(url, timeout=60000)
                await asyncio.sleep(2)  # Wacht 2 seconden tot de website geladen is
                await page.wait_for_load_state('networkidle')
            except TimeoutError:
                print(f"Timeout opgetreden bij het laden van {url}")
                continue
            except Exception as e:
                print(f"Fout opgetreden bij het laden van {url}: {e}")
                continue
            listing_elements = await page.query_selector_all('li[class*="Listing"]')

            for element in listing_elements:
                try:
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

                    listings.append({
                        'Title': title_text,
                        'Price': price_text,
                        'Description': description_text,
                        'Link': full_link,
                        'Date': date_text,
                        'Usage': usage_text,
                    })
                except Exception as e:
                    print(f"Fout bij het parseren van listing: {e}")
                    continue
            await browser.close()
    df = pd.DataFrame(listings)
    return df

if __name__ == "__main__":
    df = asyncio.run(scrape_2dehands_playwright(search_query="supreme", category="kleding-heren", max_pages=1))
    print(df)

    df.to_csv('2dehands_listings.csv', index=False)
