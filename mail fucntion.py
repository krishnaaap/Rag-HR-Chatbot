import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os  # For environment variables

def send_leave_email(user_query: str, user_id: str):
    # ⚠️ Use environment variables instead of hardcoding credentials
    sender_email = os.getenv("SENDER_EMAIL", "your_email@example.com")
    sender_password = os.getenv("SENDER_PASSWORD", "your_password_here")
    receiver_email = os.getenv("RECEIVER_EMAIL", "receiver_email@example.com")

    subject = f"Leave Request from User {user_id}"
    body = f"User has requested:\n\n{user_query}"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        print("❌ Email failed:", e)
        return False
