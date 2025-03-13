
import streamlit as st

st.set_page_config(layout="wide")

# Define navigation menu structure
nav_menu = {
    "Welcome": [
        st.Page(
            title="Welcome",
            page="laod_data/getting_data.py",

        )
    ],
    "Data": [
        st.Page(
            title="Getting Data",
            page="getting_data.py",
        ),
        st.Page(
            title="Enrich Data",
        ),
    
    ],
    

    
}

# Create navigation in sidebar and run it
nav = st.navigation(nav_menu, position="sidebar")
nav.run()