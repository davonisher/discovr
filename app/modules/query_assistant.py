import streamlit as st
from openai import OpenAI
from config import openai_api_key

client = OpenAI(api_key=openai_api_key)

def show_query_assistant() -> str:
    """
    Shows a query assistant that helps users formulate better search queries.
    Returns the chosen query.
    """
    user_input = st.text_area(
        "Describe what you're looking for:",
        placeholder="Example: I'm looking for a used iPhone in good condition, preferably an iPhone 12 or newer"
    )
    
    if st.button("Get Query Suggestions") and user_input:
        with st.spinner("Generating suggestions..."):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a search query assistant. Generate 3 effective search queries based on the user's description."
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            )
            
            suggestions = response.choices[0].message.content.split("\n")
            suggestions = [s.strip().replace("1. ", "").replace("2. ", "").replace("3. ", "") for s in suggestions if s.strip()]
            
            st.write("Suggested queries:")
            chosen_query = st.radio("Select a query:", suggestions)
            return chosen_query
            
    return "" 