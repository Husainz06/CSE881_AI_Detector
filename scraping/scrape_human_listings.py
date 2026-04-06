import os
import json
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

# Configuration
SITEMAP_PATH = 'carefarmingnetwork_sitemap.xml'
OUTPUT_FILE = 'human_listings.json'

def scrape_directory():
    # 1. Load and parse the XML sitemap
    try:
        with open(SITEMAP_PATH, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'xml')
    except FileNotFoundError:
        print(f"Error: {SITEMAP_PATH} not found.")
        return

    # Find all <loc> tags (where the URLs live)
    urls = [loc.text for loc in soup.find_all('loc')]
    print(f"Found {len(urls)} URLs in sitemap.")

    results = []

    # 2. Launch Playwright
    with sync_playwright() as p:
        # Headless=True means it runs in the background without opening a visible window
        browser = p.chromium.launch(headless=True)
        
        # Adding a standard User-Agent helps prevent basic bot detection
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # 3. Iterate through URLs
        for url in urls:
            print(f"Fetching: {url}")
            try:
                # IMPORTANT: Random delay between 2.5 and 5.5 seconds to avoid rate limits/blocks
                time.sleep(random.uniform(2.5, 5.5))
                
                # Navigate to the page and wait for the main DOM content to load
                page.goto(url, timeout=60000, wait_until="domcontentloaded")
                
                # Extract the rendered HTML
                html_content = page.content()
                page_soup = BeautifulSoup(html_content, 'html.parser')

                # Create an ID based on the URL path
                path = urlparse(url).path
                file_id = path.strip('/').split('/')[-1] or 'unknown_id'

                # Target the Title/Name
                title_tag = page_soup.find('h1', class_='entry-title')
                name = title_tag.get_text(strip=True) if title_tag else ""

                # Target the Description
                desc_div = page_soup.find('div', attrs={"data-name": "entity_field_post_content"})
                
                if desc_div:
                    # Extracts text, replacing tags with a space, and strips leading/trailing whitespace
                    description = desc_div.get_text(" ", strip=True)
                    
                    # Only save listings that actually have description text
                    if description:
                        results.append({
                            "id": file_id,
                            "url": url,
                            "name": name,
                            "description": description,
                            "label": "Human"
                        })
                        print(f"Successfully scraped: {name}")
                    else:
                        print(f"Skipped {url} - Description div found, but it was empty.")
                else:
                    print(f"Warning: Description container not found on {url}")

            except Exception as e:
                print(f"Failed to process {url}: {e}")

        # Close the browser once done
        browser.close()

    # 4. Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nDone! Saved {len(results)} records to {OUTPUT_FILE}")

if __name__ == "__main__":
    scrape_directory()