import os
import logging

def send_slack_message(insights: str, channel: str = "#discovr"):
    """
    Send market insights via Slack using the Slack SDK.
    
    Parameters:
    - insights (str): The insights text to send
    - channel (str): The Slack channel to send to (default: #discovr)
    """
    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
        
        # Get Slack token from environment variable
        slack_token = os.getenv("SLACK_BOT_TOKEN")
        
        if not slack_token:
            logging.error("❌ Slack bot token not found in environment variables.")
            return
            
        client = WebClient(token=slack_token)
        
        # Format the message
        message_body = f"""
        *Market Insights from Marktplaats Scraper*
        
        {insights}
        
        Best regards,
        Marktplaats Scraper Team
        """
        
        # Send the message
        response = client.chat_postMessage(
            channel=channel,
            text=message_body
        )
        
        logging.info("✅ Slack message sent successfully!")
        
    except ImportError:
        logging.error("❌ Slack SDK package not installed. Please install it with: pip install slack-sdk")
    except SlackApiError as e:
        logging.error(f"❌ Error sending Slack message: {str(e.response['error'])}")
    except Exception as e:
        logging.error(f"❌ Error sending Slack message: {str(e)}")

import os
import logging

def send_slack_message2(channel: str = "#discovr"):
    """
    Send market insights via Slack using the Slack SDK.
    
    Parameters:
    - channel (str): The Slack channel to send to (default: #discovr)
    """
    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
        
        # Get Slack token from environment variable
        slack_token = os.getenv("SLACK_BOT_TOKEN")
        
        if not slack_token:
            logging.error("❌ Slack bot token not found in environment variables.")
            return
            
        client = WebClient(token=slack_token)
        
        # Format the message
        message_body = f"""
        Your analysis is ready!
        """
        
        # Send the message
        response = client.chat_postMessage(
            channel=channel,
            text=message_body
        )
        
        logging.info("✅ Slack message sent successfully!")
        
    except ImportError:
        logging.error("❌ Slack SDK package not installed. Please install it with: pip install slack-sdk")
    except SlackApiError as e:
        logging.error(f"❌ Error sending Slack message: {str(e.response['error'])}")
    except Exception as e:
        logging.error(f"❌ Error sending Slack message: {str(e)}")

def handle_slack_message(message_text: str):
    """
    Handle incoming Slack messages and trigger auto search if needed.
    
    Parameters:
    - message_text (str): The text content of the Slack message
    """
    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
        from modules.auto_search_agent import run_auto_search
        
        # Get Slack token
        slack_token = os.getenv("SLACK_BOT_TOKEN")
        if not slack_token:
            logging.error("❌ Slack bot token not found in environment variables.")
            return
            
        client = WebClient(token=slack_token)

        # Send initial message that search is running
        client.chat_postMessage(
            channel="#discovr",
            text="🔄 Auto search is now running..."
        )
        
        # Run auto search with the message text as query
        search_results = run_auto_search("volkswagen golf")
        
        # Format response message
        response = f"""
        Auto search results for: volkswagen golf
        {search_results}
        """
        
        # Send results back to Slack
        client.chat_postMessage(
            channel="#discovr",
            text=response
        )
        
        logging.info("✅ Auto search completed for query: volkswagen golf")
        
    except ImportError:
        logging.error("❌ Required packages not installed")
    except SlackApiError as e:
        logging.error(f"❌ Slack API error: {str(e.response['error'])}")
    except Exception as e:
        logging.error(f"❌ Error handling Slack message: {str(e)}")
