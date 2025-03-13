import asyncio
import json
import logging
import streamlit as st
from config import deepseek_api_key
from openai import OpenAI 

client = OpenAI(base_url="https://api.deepseek.com", api_key=deepseek_api_key)
async def _deepseek_improve_query(user_description: str) -> str:
    """
    Calls Deepseek's chat API with a system + user prompt to produce a single
    improved query string. Returns that refined query.
    """
    if not user_description.strip():
        return ""

    system_prompt = (
        "You are a helpful assistant. The user will provide a description of what they want to search for.\n"
        "Your job is to output one single improved search query."
    )

    user_prompt = f"""
    Please produce a single improved or refined search query based on this description:
    '{user_description}'
    
    Return ONLY the improved query, nothing else. 
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
            )
            if response and response.choices:
                return response.choices[0].message.content.strip()
            else:
                logging.warning("No valid response from Deepseek (improve query).")
                return ""
        except Exception as e:
            logging.error(f"Exception in Deepseek call (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                return ""
    return ""

async def _deepseek_generate_three_queries(base_query: str) -> list:
    """
    Calls Deepseek to produce three alternative/refined queries
    if the user wants more suggestions beyond the first improved query.
    """
    if not base_query.strip():
        return []

    system_prompt = (
        "You are a helpful assistant. The user wants additional refined queries.\n"
        "Produce three alternative or complementary search queries based on the given one."
    )

    user_prompt = f"""
    Create three alternative search queries based on this query: '{base_query}'.
    Return them as a JSON list, e.g. ["query1", "query2", "query3"]
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
            )
            if response and response.choices:
                content_str = response.choices[0].message.content.strip()
                # We told the LLM to return them as a JSON list,
                # so we try to parse as JSON:
                try:
                    queries_list = json.loads(content_str)
                    if isinstance(queries_list, list):
                        return [q.strip() for q in queries_list]
                    else:
                        logging.warning("Deepseek returned non-list: %s", content_str)
                        return []
                except Exception:
                    # If parse fails, treat the entire content as 3 lines
                    return [line.strip() for line in content_str.split("\n") if line.strip()]
            else:
                logging.warning("No valid response from Deepseek (3 queries).")
                return []
        except Exception as e:
            logging.error(f"Exception in Deepseek call (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                return []

    return []

# -----------------------------------------------------------------------------
# 3. Main Streamlit function for the assistant UI
# -----------------------------------------------------------------------------
def show_query_assistant() -> str:
    """
    A small Streamlit UI flow that:
    1) Asks user to type a description
    2) Generates one improved query (via Deepseek)
    3) Asks if correct or user wants more suggestions
    4) If user wants more, return three refined queries
    Returns the final chosen/improved query (or empty string if user doesn't confirm).
    """
    st.write("### Deepseek Zoekassistent")
    user_description = st.text_area("Beschrijf wat je zoekt:", height=100)
    final_query = ""

    if st.button("Genereer verbeterde zoekopdracht"):
        if not user_description.strip():
            st.warning("Voer eerst een beschrijving in.")
            return ""

        with st.spinner("Deepseek is bezig met zoeken..."):
            improved_query = asyncio.run(_deepseek_improve_query(user_description))

        if not improved_query:
            st.warning("Kon geen verbeterde zoekopdracht genereren.")
            return ""

        st.markdown("**Suggestie voor zoekopdracht**:")
        st.success(improved_query)

        # Vraagt de gebruiker of ze deze query goedkeuren
        user_choice = st.radio(
            "Is deze zoekopdracht correct, of wil je meer opties?",
            ["Ja, deze is goed", "Nee, ik wil meer verfijning"]
        )

        if user_choice == "Ja, deze is goed":
            final_query = improved_query
            st.success("Zoekopdracht bevestigd!")
        else:
            st.info("We gaan verder verfijnen...")
            with st.spinner("Deepseek is bezig met meer opties..."):
                refined_queries = asyncio.run(_deepseek_generate_three_queries(improved_query))

            if not refined_queries:
                st.warning("Kon geen extra verfijningen genereren.")
                return ""

            st.markdown("**Drie mogelijke vervolg-zoekopdrachten**:")
            for i, q in enumerate(refined_queries, start=1):
                st.write(f"{i}. {q}")

            selected_query = st.selectbox("Kies er één:", refined_queries)
            if st.button("Bevestig gekozen query"):
                final_query = selected_query
                st.success(f"Gekozen zoekopdracht: {selected_query}")

    return final_query