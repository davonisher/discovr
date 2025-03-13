
import os
import io
import time
import asyncio
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from docx import Document
from docx.shared import Inches
from config import SAVED_RUNS_DIR, SAVED_RUNS_DIR_PP, PP
from modules.scrapers.scraper import scrape_marktplaats_playwright
from modules.scrapers.scraper_2dehands import scrape_2dehands_playwright
from modules.scrapers.scrape_vinted import scrape_vinted_playwright
from modules.scrapers.scrape_ebay import scrape_ebay_playwright
from modules.scrapers.scrape_catawiki import scrape_catawiki_playwright
from modules.scrapers.scraper_markt_de import scrape_markt_playwright
from modules.query_asssistant import show_query_assistant
from modules import image_search
import os
import sys

# Website Selection
website_options = {
    "Marktplaats": "scrape_marktplaats_playwright",
    "2dehands": "scrape_2dehands_playwright", 
    "Vinted": "scrape_vinted_playwright",
    "eBay": "scrape_ebay_playwright",
    "Catawiki": "scrape_catawiki_playwright",
    "Markt.de": "scrape_markt_playwright"
}

selected_websites = st.multiselect(
    "Select Websites to Scrape",
    options=list(website_options.keys()),
    default=["Marktplaats"],
    key="website_selector"
)

# Search Assistant
with st.expander("Use Search Assistant"):
    chosen_query = show_query_assistant()
    if chosen_query:
        st.session_state["assistant_query"] = chosen_query
        st.success(f"Chosen search query from assistant: {chosen_query}")

with st.expander("Use Image Search"):
    image_search.find_similar_products()

# Main scraping form
with st.form(key='scrape_form'):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # If we want to use the assistant query
        default_query = st.session_state.get("assistant_query", "")
        search_query = st.text_input("✅ Enter your search term:", value=default_query)

        # Only show category for websites that use it
        if any(site in ["Marktplaats", "2dehands"] for site in selected_websites):
            category = st.selectbox("📂 Category", get_category_options())
        else:
            category = ""  # Default empty for sites that don't use categories
            
        max_pages = st.number_input("📄 Number of pages (1-200)", min_value=1, max_value=200, value=2)
    
    with col2:
        st.empty()  # Placeholder for alignment

    submit_scrape = st.form_submit_button(label="Search")
    estimated_time = (max_pages * 7 * len(selected_websites)) / 60
    st.info(f"⏱️ Estimated wait time: {estimated_time:.1f} minutes")

if submit_scrape:
    if not search_query:
        st.error("Please enter a search query")
        return

    with st.spinner(f"🕵️‍♂️ Scraping selected websites..."):
        try:
            all_results = []
            
            # Process each website one by one
            for website in selected_websites:
                st.info(f"Scraping {website}...")
                
                # Create async function to handle scraping
                async def do_scrape():
                    if website == "Marktplaats":
                        df = await scrape_marktplaats_playwright(search_query, category, max_pages)
                        return df
                    elif website == "2dehands":
                        return await scrape_2dehands_playwright(search_query, category, max_pages)
                    elif website == "Vinted":
                        return await scrape_vinted_playwright(search_query=search_query, max_pages=max_pages)
                    elif website == "eBay":
                        return await scrape_ebay_playwright(search_query=search_query, max_pages=max_pages)
                    elif website == "Catawiki":
                        return await scrape_catawiki_playwright(search_query=search_query, max_pages=max_pages)
                    elif website == "Markt.de":
                        return await scrape_markt_playwright(search_query=search_query, max_pages=max_pages)
                
                # Run the async function
                result = asyncio.run(do_scrape())
                # Convert the result to a DataFrame if it isn't already
                if isinstance(result, pd.DataFrame):
                    result['source_file'] = website
                    all_results.append(result)
                st.success(f"Completed scraping {website}")

            # Combine all results
            df = pd.concat(all_results, ignore_index=True)

            if df.empty:
                st.error("No results found")
                return

            # Save raw data and store in databases
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            search_query_encoded = sanitize_filename_component(search_query)
            
            # Create filenames for each selected website
            for website in selected_websites:
                csv_filename = f"Raw_{website}_{search_query_encoded}_{timestamp}.csv"
                csv_path = os.path.join(SAVED_RUNS_DIR, csv_filename)

                # Filter data for current website
                website_df = df[df['source_file'] == website].copy()
                
                # Add metadata to the DataFrame before saving
                website_df['search_query'] = search_query
                website_df['category'] = category
                website_df['website'] = website
                
                # Use redundant storage to save to both PostgreSQL and Supabase
                try:
                    st.info(f"Saving data to databases for {website}...")
                    result = save_raw_data_redundant(website_df, source_file=csv_filename)
                    
                    if result["overall_success"]:
                        if result["postgres"]["success"] and result["supabase"]["success"]:
                            st.success(f"✅ Data successfully stored in both PostgreSQL and Supabase for {website}")
                        elif result["postgres"]["success"]:
                            st.warning(f"✅ Data stored in PostgreSQL but not Supabase for {website}")
                        elif result["supabase"]["success"]:
                            st.warning(f"✅ Data stored in Supabase but not PostgreSQL for {website}")
                    else:
                        st.error(f"❌ Failed to store data in any database for {website}")
                        
                        # Save to CSV as final fallback
                        try:
                            website_df.to_csv(csv_path, index=False)
                            st.info(f"📁 Data saved to CSV for {website} at: {csv_path}")
                        except Exception as csv_e:
                            st.error(f"❌ Error saving data to CSV for {website}: {csv_e}")
                except Exception as e:
                    st.error(f"❌ Error during database operations: {e}")
                    
                    # Save to CSV as fallback
                    try:
                        website_df.to_csv(csv_path, index=False)
                        st.info(f"📁 Data saved to CSV for {website} at: {csv_path}")
                    except Exception as csv_e:
                        st.error(f"❌ Error saving data to CSV for {website}: {csv_e}")

            # Store in session state
            st.session_state["raw_df"] = df
            st.session_state["preprocessed_df"] = df

            # Display results
            st.subheader("Search Results")
            st.dataframe(df)

            # Start the file watcher
            start_file_watcher()

        except Exception as e:
            st.error(f"❌ Error during scraping: {e}")
            return