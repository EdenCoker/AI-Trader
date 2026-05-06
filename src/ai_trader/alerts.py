from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


def send_alert(subject: str, body: str) -> bool:
    """Send an optional email alert configured entirely through environment variables."""

    recipient = os.getenv("ALERT_EMAIL")
    smtp_host = os.getenv("SMTP_HOST")
    if not recipient or not smtp_host:
        return False

    sender = os.getenv("SMTP_FROM") or os.getenv("SMTP_USERNAME") or recipient
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)

    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    use_tls = os.getenv("SMTP_TLS", "true").casefold() not in {"0", "false", "no", "off"}

    with smtplib.SMTP(smtp_host, port, timeout=20) as server:
        if use_tls:
            server.starttls()
        if username and password:
            server.login(username, password)
        server.send_message(message)
    return True
