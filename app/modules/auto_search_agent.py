import asyncio
import logging
import os
import sys
import time
from typing import List, Dict, Tuple
import streamlit as st

# Fix path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main_local import orchestrate_scraper, preprocess_enriched_df
from modules.db.database import store_raw_data, store_preprocessed_data
from modules.slack import send_slack_message2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('auto_search.log')
    ]
)

# Define your search configurations
DEFAULT_CONFIGS = [
    {"query": "bowers wilkins", "category": "audio-tv-en-foto", "pages": 5},
    {"query": "bang olufsen", "category": "audio-tv-en-foto", "pages": 5},
    {"query": "dali speakers", "category": "audio-tv-en-foto", "pages": 5},
    {"query": "macbook pro", "category": "computers-en-software", "pages": 5},
    {"query": "ipad pro", "category": "computers-en-software", "pages": 5},
    {"query": "samsung galaxy", "category": "telecommunicatie", "pages": 5},
]

async def run_single_search(config: Dict) -> Tuple[bool, str]:
    """Run a single search configuration and process the results."""
    try:
        # Extract config
        query = config["query"]
        category = config["category"]
        pages = config["pages"]
        
        logging.info(f"Starting search for: {query} in {category}")
        st.info(f"🔍 Searching for: {query} in {category}")
        
        # 1. Scrape data
        df_raw = await orchestrate_scraper(query, category, pages)
        if df_raw.empty:
            msg = f"No results found for {query} in {category}"
            logging.warning(msg)
            st.warning(msg)
            return False, msg
            
        # 2. Store raw data
        timestamp = int(time.time())
        raw_filename = f"Raw_{query.replace(' ', '_')}_{category}_{pages}_{timestamp}.csv"
        store_raw_data(df_raw, source_file=raw_filename)
        st.success(f"✅ Stored raw data for {query}")
        
        # 3. Preprocess data
        df_preprocessed = preprocess_enriched_df(df_raw)
        if not df_preprocessed.empty:
            # 4. Store preprocessed data
            pp_filename = f"Preprocessed_{query.replace(' ', '_')}_{category}_{timestamp}.csv"
            store_preprocessed_data(df_preprocessed, source_file=pp_filename)
            st.success(f"✅ Stored preprocessed data for {query}")
            
        return True, f"Successfully processed {query}"
        
    except Exception as e:
        error_msg = f"Error processing {config['query']}: {e}"
        logging.error(error_msg)
        st.error(error_msg)
        return False, str(e)

async def run_auto_searches(configs=None):
    """Run all configured searches sequentially."""
    if configs is None:
        configs = DEFAULT_CONFIGS
        
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, config in enumerate(configs):
        status_text.text(f"Processing {config['query']}...")
        success, message = await run_single_search(config)
        results.append({
            "query": config["query"],
            "category": config["category"],
            "success": success,
            "message": message
        })
        
        # Update progress
        progress = (idx + 1) / len(configs)
        progress_bar.progress(progress)
        
        # Wait a bit between searches
        await asyncio.sleep(5)
        
    # Log final results
    summary = "=== Search Results Summary ===\n"
    for result in results:
        status = "✅" if result["success"] else "❌"
        summary += f"{status} {result['query']}: {result['message']}\n"
    
    logging.info(summary)
    st.success(summary)
    
    # Send Slack notification when done
    try:
        send_slack_message2()
        st.success("✅ Sent Slack notification")
    except Exception as e:
        st.error(f"❌ Failed to send Slack notification: {e}")

def run_auto_search():
    """Streamlit interface for auto search."""
    st.subheader("🤖 Auto Search Configuration")
    
    # Allow users to modify existing configs or add new ones
    configs = st.session_state.get("search_configs", DEFAULT_CONFIGS.copy())
    
    # Add new search
    with st.expander("Add New Search"):
        col1, col2, col3 = st.columns([2,2,1])
        with col1:
            new_query = st.text_input("Search Query")
        with col2:
            new_category = st.text_input("Category")
        with col3:
            new_pages = st.number_input("Pages", min_value=1, max_value=50, value=5)
            
        if st.button("Add Search"):
            configs.append({
                "query": new_query,
                "category": new_category,
                "pages": new_pages
            })
            st.session_state["search_configs"] = configs
            st.success("✅ Search added")
    
    # Show current configs
    st.write("Current Search Configurations:")
    for idx, config in enumerate(configs):
        st.write(f"{idx+1}. {config['query']} in {config['category']} ({config['pages']} pages)")
    
    # Run button
    if st.button("🚀 Run Auto Search"):
        asyncio.run(run_auto_searches(configs))

if __name__ == "__main__":
    try:
        asyncio.run(run_auto_searches())
    except KeyboardInterrupt:
        logging.info("Auto-search stopped by user")
    except Exception as e:
        logging.error(f"Auto-search failed: {e}")