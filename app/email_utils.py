import time
import logging
import requests
import streamlit as st
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from pprint import pprint
from app.config import brevo_api_key

def generate_email_html(insights: str) -> str:
    """
    Generates a styled HTML content for the email using the insights text.
    """
    html = f"""
    <html>
    <head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333333;
        }}
        h2 {{
            color: #4CAF50;
        }}
        p {{
            margin-bottom: 10px;
        }}
        ul {{
            margin-left: 20px;
        }}
    </style>
    </head>
    <body>
    <h2>Market Insights from Marktplaats Scraper</h2>
    <p>Congratulations! Here are the latest market insights:</p>
    {insights}
    <p>Best regards,<br>Marktplaats Scraper Team</p>
    </body>
    </html>
    """
    return html

def send_email_insights(insights: str, recipient_email: str, recipient_name: str):
    """
    Send market insights via a transactional email using Brevo's API.
    
    Parameters:
    - insights (str): The insights text to send.
    - recipient_email (str): Recipient's email address.
    - recipient_name (str): Recipient's name.
    """
    # Check if insights are empty
    if not insights.strip():
        st.warning("No insights available to send in the email.")
        return

    # Define the API endpoint
    url = "https://api.brevo.com/v3/smtp/email"

    # Define the sender
    sender = {
        "name": "Marktplaats Scraper",
        "email": "davidkakaniss@gmail.com"  # Ensure this email is verified in Brevo
    }

    # Define the email content
    html_content = generate_email_html(insights)

    # Define the payload
    payload = {
        "sender": sender,
        "to": [
            {
                "email": recipient_email,
                "name": recipient_name
            }
        ],
        "subject": "Market Insights from Marktplaats Scraper",
        "htmlContent": html_content
    }

    # Define the headers
    headers = {
        "accept": "application/json",
        "api-key": brevo_api_key,
        "content-type": "application/json"
    }

    # Send the POST request
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        st.success("Email sent successfully!")
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err} - {response.text}")
    except Exception as err:
        st.error(f"An error occurred: {err}")

def send_email_campaign(insights: str):
    """
    Send market insights via email campaign using Brevo API.
    """
    # Instantiate the client with proper configuration
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key['api-key'] = brevo_api_key
    api_instance = sib_api_v3_sdk.EmailCampaignsApi(sib_api_v3_sdk.ApiClient(configuration))

    # Define the campaign settings
    email_campaigns = sib_api_v3_sdk.CreateEmailCampaign(
        name="Market Insights Campaign",
        subject="Market Insights from Marktplaats Scraper",
        sender={"name": "Marktplaats Scraper", "email": "davidkakaniss@gmail.com"},
        type="classic",
        # Content that will be sent
        html_content=generate_email_html(insights),
        # Select the recipients
        recipients={"listIds": [2, 7]},
        # Schedule the sending immediately
        scheduled_at=time.strftime("%Y-%m-%d %H:%M:%S")
    )

    # Make the call to the client
    try:
        api_response = api_instance.create_email_campaign(email_campaigns)
        pprint(api_response)
        st.success("Email campaign sent successfully!")
    except ApiException as e:
        st.error(f"Exception when calling EmailCampaignsApi->create_email_campaign: {e}")

def send_email_with_attachment(insights: str, recipient_email: str, recipient_name: str, attachment_path: str = None):
    """
    Send market insights via email with optional attachment using Brevo's API.
    
    Parameters:
    - insights (str): The insights text to send.
    - recipient_email (str): Recipient's email address.
    - recipient_name (str): Recipient's name.
    - attachment_path (str, optional): Path to the file to attach.
    """
    # Check if insights are empty
    if not insights.strip():
        st.warning("No insights available to send in the email.")
        return

    # Define the API endpoint
    url = "https://api.brevo.com/v3/smtp/email"

    # Define the sender
    sender = {
        "name": "Marktplaats Scraper",
        "email": "davidkakaniss@gmail.com"
    }

    # Define the email content
    html_content = generate_email_html(insights)

    # Define the payload
    payload = {
        "sender": sender,
        "to": [
            {
                "email": recipient_email,
                "name": recipient_name
            }
        ],
        "subject": "Market Insights from Marktplaats Scraper",
        "htmlContent": html_content
    }

    # Add attachment if provided
    if attachment_path:
        try:
            with open(attachment_path, "rb") as f:
                import base64
                content = base64.b64encode(f.read()).decode('utf-8')
                payload["attachment"] = [
                    {
                        "content": content,
                        "name": attachment_path.split("/")[-1]
                    }
                ]
        except Exception as e:
            st.error(f"Error reading attachment: {e}")
            return

    # Define the headers
    headers = {
        "accept": "application/json",
        "api-key": brevo_api_key,
        "content-type": "application/json"
    }

    # Send the POST request
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        st.success("Email sent successfully!")
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err} - {response.text}")
    except Exception as err:
        st.error(f"An error occurred: {err}") 