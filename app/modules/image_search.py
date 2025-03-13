import streamlit as st
import pandas as pd
import requests
import json
from io import BytesIO
from PIL import Image
from price_parser import Price
import os
import base64
from groq import Groq, GroqError

groq_api_key = os.getenv("GROQ_API_KEY")

# Groq API Constants
client = None
if groq_api_key:
    try:
        client = Groq(api_key=groq_api_key)
    except GroqError as e:
        st.error(f"Failed to initialize Groq client: {str(e)}")

def encode_image(image: Image) -> str:
    """
    Encode PIL Image to base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")  # Save as JPEG format
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def classify_image_with_groq(image: Image) -> dict:
    """
    Send an image to Groq's vision model and get product classification.

    Args:
        image (Image): A PIL Image object.

    Returns:
        dict: A dictionary with predicted categories and details.
    """
    if not client:
        return {"error": "Groq client is not initialized due to missing or invalid API key."}

    base64_image = encode_image(image)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": "Classify this product and provide its key characteristics."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            model="llama-3.2-90b-vision-preview",
            tools=[{
                "type": "function",
                "function": {
                    "name": "classify_product",
                    "description": "Classify product details from image",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Brand": {
                                "type": "string",
                                "description": "Brand name of the product"
                            },
                            "Model": {
                                "type": "string",
                                "description": "Model name/number of the product"
                            },
                            "Category": {
                                "type": "string",
                                "description": "Product category"
                            },
                            "Condition": {
                                "type": "string",
                                "description": "Product condition"
                            }
                        },
                        "required": ["Brand", "Model", "Category", "Condition"]
                    }
                }
            }],
            tool_choice="auto"
        )
        
        return json.loads(chat_completion.choices[0].message.tool_calls[0].function.arguments)
    except Exception as e:
        return {"error": f"API error: {str(e)}"}

def search_similar_products(classification: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Search the dataset for products matching the classification.

    Args:
        classification (dict): The vision model output.
        df (pd.DataFrame): The existing product database.

    Returns:
        pd.DataFrame: A filtered DataFrame with matching products.
    """
    if "Brand" in classification and "Model" in classification:
        filtered_df = df[
            (df["Brand"].str.contains(classification["Brand"], case=False, na=False)) &
            (df["Model"].str.contains(classification["Model"], case=False, na=False))
        ]
    else:
        filtered_df = df

    return filtered_df

def find_similar_products():
    """
    Streamlit UI function to upload an image, classify the product using Groq AI,
    and search the database for similar products.
    """
    st.header("🔍 Search for similar products via image")

    uploaded_image = st.file_uploader("📸 Upload a product image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Show the image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded image", use_column_width=True)

        # Classify image with vision model
        if st.button("🔎 Classify product"):
            with st.spinner("Analyzing with AI..."):
                classification = classify_image_with_groq(image)
                if "error" in classification:
                    st.error(classification["error"])
                else:
                    st.success("✅ Classification complete!")
                    st.json(classification)

                    # Search database
                    if "enriched_df" in st.session_state:
                        df = st.session_state["enriched_df"]
                        results = search_similar_products(classification, df)

                        if not results.empty:
                            st.success(f"✅ Found {len(results)} similar products!")
                            st.dataframe(results)
                        else:
                            st.warning("⚠️ No similar products found.")
                    else:
                        st.warning("⚠️ No database loaded. Please enrich your data first.")
