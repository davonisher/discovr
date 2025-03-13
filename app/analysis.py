import asyncio
import json
import logging
import requests
import pandas as pd
import streamlit as st
from app.config import openai_api_key
from app.utils import clean_price

import nest_asyncio

nest_asyncio.apply()

async def price_trend_analysis(df: pd.DataFrame) -> str:
    """
    Analyzes pricing trends for different brands and models over time.
    """
    if df.empty:
        return "No data available for price trend analysis."

    # Ensure CleanedPrice and Date are present
    if 'CleanedPrice' not in df.columns:
        df['CleanedPrice'] = df['Price'].apply(clean_price)
    if 'Date' not in df.columns:
        st.warning("No Date column found.")
        return "No date information available for trend analysis."

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Sort by Date
    df = df.sort_values('Date')

    # Analyze trends for top 5 brands
    top_brands = df['Brand'].value_counts().head(5).index.tolist()

    trend_summary = ""

    for brand in top_brands:
        brand_df = df[df['Brand'] == brand]
        if brand_df['Date'].isnull().all():
            trend = "No date information available."
        else:
            brand_df = brand_df.dropna(subset=['Date'])
            brand_df = brand_df.set_index('Date').resample('M')['CleanedPrice'].mean().reset_index()
            price_diff = brand_df['CleanedPrice'].iloc[-1] - brand_df['CleanedPrice'].iloc[0]
            if price_diff > 0:
                trend = "Increasing"
            elif price_diff < 0:
                trend = "Decreasing"
            else:
                trend = "Stable"
        trend_summary += f"**{brand}**: {trend}\n\n"

    return trend_summary

async def market_insights_analysis(df: pd.DataFrame) -> str:
    """
    Provides comprehensive market insights based on the scraped data.
    """
    if df.empty:
        return "No data available for market insights analysis."

    # Prepare summary
    total_listings = len(df)
    avg_price = df['CleanedPrice'].mean() if 'CleanedPrice' in df.columns else 0.0
    median_price = df['CleanedPrice'].median() if 'CleanedPrice' in df.columns else 0.0
    top_brands = df['Brand'].value_counts().head(5).to_dict()
    top_models = df['Model'].value_counts().head(5).to_dict()
    usage_conditions = df['Usage'].value_counts().to_dict()

    summary = f"""
    You are an AI assistant specialized in market analysis. Based on the provided data, please provide comprehensive market insights.

    **Data Summary:**
    - Total Listings: {total_listings}
    - Average Price: €{avg_price:.2f}
    - Median Price: €{median_price:.2f}
    - Top Brands: {', '.join([f"{brand} ({count})" for brand, count in top_brands.items()])}
    - Top Models: {', '.join([f"{model} ({count})" for model, count in top_models.items()])}
    - Usage Conditions: {', '.join([f"{usage} ({count})" for usage, count in usage_conditions.items()])}

    **Please provide your market insights below:**
    """

    json_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant specialized in market analysis and providing comprehensive insights."
                )
            },
            {
                "role": "user",
                "content": summary
            }
        ],
        "temperature": 0.7,
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
                return response_data['choices'][0]['message']['content']
            elif response.status_code == 429:
                logging.warning(f"Rate limit hit. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                logging.warning(f"Got HTTP {response.status_code}. Body: {response.text}")
                return "Error generating market insights."
        except Exception as e:
            logging.error(f"Exception while calling LLM: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2

    return "Failed to generate market insights after multiple retries."

async def sentiment_analysis_agent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs sentiment analysis on product listings.
    """
    if df.empty:
        return df

    async def analyze_sentiment(text: str) -> tuple:
        prompt = f"""
        Analyze the sentiment of this product listing text. Consider factors like:
        - Product condition
        - Seller's tone
        - Description completeness
        - Price fairness (if mentioned)

        Text: "{text}"

        Return a tuple of (sentiment, score) where:
        - sentiment is one of: "Positive", "Neutral", or "Negative"
        - score is a float between -1.0 (most negative) and 1.0 (most positive)
        """

        json_data = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a sentiment analysis expert."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
        }

        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=json_data,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']
                # Parse the response - expecting format like: ('Positive', 0.8)
                sentiment, score = eval(result)
                return sentiment, float(score)
            else:
                return "Neutral", 0.0
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {e}")
            return "Neutral", 0.0

    # Process each row
    sentiments = []
    scores = []
    
    for _, row in df.iterrows():
        text = f"{row['Title']} {row['Description']}"
        sentiment, score = await analyze_sentiment(text)
        sentiments.append(sentiment)
        scores.append(score)

    # Add results to DataFrame
    df['Sentiment'] = sentiments
    df['SentimentScore'] = scores

    return df

async def best_deal_recommendation(df: pd.DataFrame) -> str:
    """
    Recommends the best deals based on price, condition, and sentiment.
    """
    if df.empty:
        return "No data available for deal recommendations."

    # Ensure we have cleaned prices
    if 'CleanedPrice' not in df.columns:
        df['CleanedPrice'] = df['Price'].apply(clean_price)

    # Sort by price and get top 5 cheapest listings
    cheapest = df.nsmallest(5, 'CleanedPrice')

    # Format the deals for the LLM
    deals_text = ""
    for _, deal in cheapest.iterrows():
        deals_text += f"""
        Title: {deal['Title']}
        Price: {deal['Price']}
        Usage: {deal['Usage']}
        Description: {deal['Description']}
        Link: {deal['Link']}
        """

    prompt = f"""
    Analyze these marketplace listings and recommend the best deals. Consider:
    - Price relative to market
    - Item condition
    - Description completeness
    - Seller credibility signals

    Listings:
    {deals_text}

    Provide a concise analysis of the top 2-3 best deals, explaining why they're good value.
    """

    json_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert at finding good deals in marketplace listings."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
            timeout=60
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "Error generating deal recommendations."
    except Exception as e:
        logging.error(f"Error in deal recommendation: {e}")
        return "Failed to generate deal recommendations."

async def market_demand_forecasting(df: pd.DataFrame) -> str:
    """
    Forecasts market demand trends based on historical data.
    """
    if df.empty:
        return "No data available for demand forecasting."

    # Convert dates and sort
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')

    # Calculate basic metrics
    listings_per_day = df.groupby('Date').size().mean()
    price_trend = df.groupby('Date')['CleanedPrice'].mean()
    price_change = (price_trend.iloc[-1] - price_trend.iloc[0]) / price_trend.iloc[0] * 100

    # Get top brands and their listing counts
    brand_demand = df['Brand'].value_counts().head(5)

    prompt = f"""
    Based on the marketplace data, provide demand forecasting insights:

    Metrics:
    - Average Daily Listings: {listings_per_day:.1f}
    - Price Trend: {price_change:.1f}% change over period
    - Most Popular Brands: {', '.join([f"{brand} ({count})" for brand, count in brand_demand.items()])}

    Analyze these metrics and provide:
    1. Short-term demand forecast
    2. Price trend predictions
    3. Brand popularity predictions
    4. Recommendations for buyers/sellers
    """

    json_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert market analyst specializing in demand forecasting."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
            timeout=60
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "Error generating demand forecast."
    except Exception as e:
        logging.error(f"Error in demand forecasting: {e}")
        return "Failed to generate demand forecast."

async def run_agent_analysis(df: pd.DataFrame) -> str:
    """
    Comprehensive agent analysis combining multiple insights.
    """
    insights = ""

    # Best Deal Recommendation
    best_deal = await best_deal_recommendation(df)
    insights += f"**Best Deal Recommendation:**\n{best_deal}\n\n"

    # Price Trend Analysis
    price_trend = await price_trend_analysis(df)
    insights += f"**Price Trend Analysis:**\n{price_trend}\n\n"

    # Market Insights
    market_insights = await market_insights_analysis(df)
    insights += f"**Market Insights:**\n{market_insights}\n\n"

    # Market Demand Forecasting
    demand_forecasting = await market_demand_forecasting(df)
    insights += f"**Market Demand Forecasting:**\n{demand_forecasting}\n\n"

    return insights

async def detailed_comparative_analysis(df: pd.DataFrame) -> str:
    """
    Uses an LLM to compare similar listings and highlight the best options based on multiple criteria.
    """
    if df.empty:
        return "No data available for comparative analysis."

    # Ensure CleanedPrice is present
    if 'CleanedPrice' not in df.columns:
        df['CleanedPrice'] = df['Price'].apply(clean_price)

    # Select top 10 cheapest listings as examples
    sample_df = df.nsmallest(10, 'CleanedPrice')

    # Convert sample data to a readable format
    listings_info = ""
    for idx, row in sample_df.iterrows():
        listings_info += f"""
        **Listing {idx + 1}:**
        - **Title:** {row['Title']}
        - **Price:** {row['Price']}
        - **Brand:** {row['Brand']}
        - **Model:** {row['Model']}
        - **Usage:** {row['Usage']}
        - **Link:** {row['Link']}
        - **Date:** {row['Date']}
        - **Sentiment:** {row.get('Sentiment', 'Neutral')} (Score: {row.get('SentimentScore', 0.0)})
        """

    prompt = f"""
    You are an expert market analyst. Compare the following marketplace listings and recommend the best options based on price, brand reputation, model popularity, usage condition, and sentiment score.

    {listings_info}

    **Provide your analysis and recommendations below:**
    """

    json_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant specialized in market analysis and comparative evaluations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
            timeout=60
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "Error generating comparative analysis."
    except Exception as e:
        logging.error(f"Error in comparative analysis: {e}")
        return "Failed to generate comparative analysis." 