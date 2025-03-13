
from sympy import Rem
from tblib import Code
from twilio.rest import Client


client = Client(account_sid, auth_token)

message = client.messages.create(
  from_='whatsapp:+14155238886',
  body="""',Title,Price,Description,Link,Date,Usage,Brand,Model,ClassifiedDate,embedding,token_count,top_5_uses,top_5_scores
0,SALE! Refurbished laptops vanaf 199wn', 'Dell', 'LENOVO']","[0.7478221654891968, 0.6742018461227417, 0.6682673096656799, 0.6581084728240967, 0.6511214375495911]"
4,"Laptop: Lenovo ThinkPad T15 Gen """,
  to='whatsapp:+31636136335'
)

print(message.sid)




def send_whatsapp_message(insights: str, recipient_number: str):
    """
    Send market insights via WhatsApp using Twilio's API.
    
    Parameters:
    - insights (str): The insights text to send
    - recipient_number (str): Recipient's phone number in format: +[country_code][number]
    """
    try:
        from twilio.rest import Client
        
        # Get Twilio credentials from environment variables
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        
        if not account_sid or not auth_token:
            st.error("❌ Twilio credentials not found in environment variables.")
            return
            
        client = Client(account_sid, auth_token)
        
        # Format the message
        message_body = f"""
        *Market Insights from Marktplaats Scraper*
        
        {insights}
        
        Best regards,
        Marktplaats Scraper Team
        """
        
        # Send the message
        message = client.messages.create(
            from_='whatsapp:+14155238886',  # Your Twilio WhatsApp number
            body=message_body,
            to=f'whatsapp:{recipient_number}'
        )
        
        st.success("✅ WhatsApp message sent successfully!")
        
    except ImportError:
        st.error("❌ Twilio package not installed. Please install it with: pip install twilio")
    except Exception as e:
        st.error(f"❌ Error sending WhatsApp message: {str(e)}")

