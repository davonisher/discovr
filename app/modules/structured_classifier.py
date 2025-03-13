


### TO DO 
### Date moet daadwerkelijk datum zijn
## merk moet geen merk zijn als er niks over wordt gemeld
### Alleen analyse op Zo goed als nieuw en Gebruikt
### als er een website bij staat is het product nieuw dus dan niet categoriseren
###auto pagina is anders

import pandas as pd
import json
import asyncio
import aiohttp
import time
import tiktoken
import random
import logging
import openai
import pdb  # Import the pdb module for debugging
import os
# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set your rate limits (adjust according to your actual limits)
MAX_REQUESTS_PER_MINUTE = 50  # Maximum requests per minute
MAX_TOKENS_PER_MINUTE = 125000  # Maximum tokens per minute
MAX_CONCURRENT_REQUESTS = 10  # Maximum number of concurrent requests
TOKEN_ENCODING_NAME = "cl100k_base"  # Token encoding for counting

# Semaphore to limit the number of concurrent requests
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Token bucket to track token usage
tokens_lock = asyncio.Lock()
tokens_available = MAX_TOKENS_PER_MINUTE
last_token_check_time = time.time()

# Function to count tokens in the prompt
def count_tokens(prompt, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    # Convert prompt to string if it's not already
    if not isinstance(prompt, str):
        prompt = str(prompt)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens

# Asynchronous function to process each request
async def classify_listing_with_llm(title: str, description: str, date_str: str) -> dict:
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

    # Estimate tokens (adjust max_tokens as needed)
    max_tokens = 1000
    num_tokens = count_tokens(prompt, TOKEN_ENCODING_NAME) + max_tokens

    # Wait for semaphore to ensure we don't exceed MAX_CONCURRENT_REQUESTS
    async with semaphore:
        # Rate limiting: ensure we don't exceed tokens per minute
        global tokens_available, last_token_check_time
        async with tokens_lock:
            current_time = time.time()
            elapsed = current_time - last_token_check_time
            tokens_available += (MAX_TOKENS_PER_MINUTE / 60) * elapsed
            tokens_available = min(tokens_available, MAX_TOKENS_PER_MINUTE)
            last_token_check_time = current_time

            if tokens_available < num_tokens:
                # Need to wait for tokens to replenish
                wait_time = (num_tokens - tokens_available) / (MAX_TOKENS_PER_MINUTE / 60)
                minutes, seconds = divmod(wait_time, 60)
                logging.info(f"Waiting for {int(minutes)} minutes and {seconds:.2f} seconds due to token limit.")
                await asyncio.sleep(wait_time)
                tokens_available = 0
            else:
                tokens_available -= num_tokens

        # Prepare the request payload
        json_data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an AI model tasked with extracting detailed product information from a marketplace listing. "
                        "The extracted information must include: brand, model, and date. "
                        "The response should be structured as JSON."
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
                            "brand": {
                                "type": "string",
                                "description": "The brand or manufacturer of the product."
                            },
                            "model": {
                                "type": "string",
                                "description": "The specific model name or identifier of the product."
                            },
                            "date": {
                                "type": "string",
                                "description": "The date associated with the listing."
                            }
                        },
                        "required": ["brand", "model", "date"],
                        "additionalProperties": False
                    }
                }
            ],
            "function_call": {"name": "extract_product_info"},
            "max_tokens": max_tokens
        }

        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json"
        }

        # Implement exponential backoff for retries
        retry_delay = 1  # Initial delay in seconds
        max_retries = 5

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=json_data,
                        timeout=60
                    ) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            arguments = response_data['choices'][0]['message']['function_call']['arguments']
                            result_dict = json.loads(arguments)

                            # Eventueel data opschonen (bv. als brand=None => "Unknown").
                            #    Zorg dat date uit de LLM niet je eigen date_str overschrijft, tenzij je dat wilt.
                            final_dict = {
                                "brand": result_dict.get("brand", "Unknown"),
                                "model": result_dict.get("model", "Unknown"),
                                "date": result_dict.get("date", date_str),
                                "raw_response": response_data['choices'][0]['message']['content']  # bewaar het antwoord voor debugging
                            }

                            # Debug: Print the final dictionary
                            print(f"Final dictionary: {final_dict}")

                            # Add a breakpoint for debugging
                            pdb.set_trace()

                            return final_dict
                        elif response.status == 429:
                            # Rate limit exceeded
                            logging.warning(f"Rate limit hit. Retrying after {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            # Other HTTP errors
                            error_text = await response.text()
                            logging.error(f"Error {response.status}: {error_text}")
                            return None
            except Exception as e:
                logging.error(f"Exception on attempt {attempt + 1}: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
                continue
        logging.error(f"Failed to classify listing after {max_retries} attempts.")
        return None
