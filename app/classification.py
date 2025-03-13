import asyncio
import json
import logging
import requests
import pandas as pd
from openai import OpenAI
from app.config import deepseek_api_key, openai_api_key

# Initialize the DeepSeek client
client = OpenAI(base_url="https://api.deepseek.com", api_key=deepseek_api_key)

async def classify_listing_with_llm2(title: str, description: str, date_str: str) -> dict:
    """
    Uses the DeepSeek API (with 'deepseek-chat' model) to get structured JSON
    for brand, model and date.
    """
    user_prompt = f"""
    You are a product classifier. You receive a Title, Description, and Date from a marketplace listing.
    You must extract:
    - brand (the manufacturer or brand name)
    - model (the specific model name or identifier)
    - date (the date from the listing)

    Return the result *strictly* as JSON with keys "brand", "model", and "date".
    If you are uncertain, use "Unknown" for brand or model.

    Title: "{title}"
    Description: "{description}"
    Date: "{date_str}"

    JSON output:
    """

    system_prompt = (
        "You are an AI model tasked with extracting product information from a listing. "
        "The extracted info must be brand, model, and date, in JSON."
    )
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
                response_format={"type": "json_object"}
            )

            if response and response.choices:
                content_str = response.choices[0].message.content
                result_dict = json.loads(content_str)

                final_dict = {
                    "brand": result_dict.get("brand", "Unknown"),
                    "model": result_dict.get("model", "Unknown"),
                    "date":  result_dict.get("date", date_str),
                }
                return final_dict
            else:
                logging.warning("No valid response received from DeepSeek.")
                return None

        except Exception as e:
            logging.error(f"Exception in DeepSeek call (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                return None

    return None

async def classify_listing_with_llm(title: str, description: str, date_str: str) -> dict:
    """
    Uses GPT-4 or another large language model to get brand, model, and date.
    """
    prompt = f"""
    You are a product classifier. You receive a Title, Description, and Date from a marketplace listing.
    You must extract:
    - brand (the manufacturer or brand name)
    - model (the specific model name or identifier)
    - date (the date from the listing)

    Return the result *strictly* as JSON with keys "brand", "model", and "date".
    If you are uncertain, use "Unknown" for brand or model.

    Title: "{title}"
    Description: "{description}"
    Date: "{date_str}"

    JSON output:
    """

    json_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an AI model tasked with extracting product information from a listing. "
                    "The extracted info must be brand, model, and date, in JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "functions": [
            {
                "name": "extract_product_info",
                "description": "Extracts product information from a marketplace listing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "brand": {"type": "string"},
                        "model": {"type": "string"},
                        "date": {"type": "string"}
                    },
                    "required": ["brand", "model", "date"]
                }
            }
        ],
        "function_call": {"name": "extract_product_info"},
    }

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    max_retries = 3
    retry_delay = 1

    for _ in range(max_retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=json_data,
                timeout=60
            )
            if response.status_code == 200:
                response_data = response.json()
                arguments = response_data['choices'][0]['message']['function_call']['arguments']
                result_dict = json.loads(arguments)

                final_dict = {
                    "brand": result_dict.get("brand", "Unknown"),
                    "model": result_dict.get("model", "Unknown"),
                    "date": result_dict.get("date", date_str),
                }
                return final_dict

            elif response.status_code == 429:
                logging.warning(f"Rate limit hit. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                logging.warning(f"Got HTTP {response.status_code}. Body: {response.text}")
                return None

        except Exception as e:
            logging.error(f"Exception while calling LLM: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2

    return None

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the DataFrame by classifying each listing using the LLM.
    
    Parameters:
    - df (pd.DataFrame): The raw scraped DataFrame.
    
    Returns:
    - pd.DataFrame: The enriched DataFrame with additional classification columns.
    """
    async def classify_row_async(row):
        title = row.get("Title", "")
        description = row.get("Description", "")
        date_str = row.get("Date", "")
        
        # Call the asynchronous classification function
        llm_result = await classify_listing_with_llm2(title, description, date_str)
        if llm_result is None:
            return {
                "Brand": "Unknown",
                "Model": "Unknown",
                "ClassifiedDate": date_str
            }
        else:
            return {
                "Brand": llm_result.get("brand", "Unknown"),
                "Model": llm_result.get("model", "Unknown"),
                "ClassifiedDate": llm_result.get("date", date_str)
            }
    
    # Run the enrichment asynchronously
    async def enrich_async():
        enriched_data = await asyncio.gather(*[classify_row_async(row) for _, row in df.iterrows()])
        return pd.DataFrame(enriched_data)
    
    # Execute the async enrichment
    enriched_df = asyncio.run(enrich_async())
    
    # Concatenate the classification results to the original DataFrame
    enriched_df = pd.concat([df.reset_index(drop=True), enriched_df.reset_index(drop=True)], axis=1)
    
    return enriched_df 