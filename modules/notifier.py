import smtplib
from email.mime.text import MIMEText

def send_email_notification(to_email, subject, body):
    """
    Eenvoudig voorbeeld via SMTP. Pas dit aan naar je eigen mailserver/credentials.
    """
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = "no-reply@my-marketplace-scraper.com"
    msg['To'] = to_email

    try:
        with smtplib.SMTP("smtp.yourprovider.com", 587) as server:
            server.starttls()
            server.login("jouw_gebruikersnaam", "jouw_wachtwoord")
            server.send_message(msg)
        print(f"Mail naar {to_email} verzonden.")
    except Exception as e:
        print(f"Fout bij mail: {str(e)}")
