#!/usr/bin/env python3
import argparse
import requests
import io

def send_to_discord(webhook_url: str, content: str):
    """
    Sendet Inhalt an Discord. Wenn der Text länger als 2000 Zeichen ist,
    wird automatisch eine Datei angehängt.
    """
    if not webhook_url:
        print("Kein Webhook angegeben")
        return

    # Discord hat ein Limit von 2000 Zeichen (inkl. Codeblock)
    prefix = "```\n"
    suffix = "\n```"
    full_message = f"{prefix}{content}{suffix}"

    if len(full_message) <= 2000:
        # Direkt als Nachricht senden
        response = requests.post(webhook_url, json={"content": full_message}, timeout=20)
    else:
        # Als Datei senden
        file_bytes = io.BytesIO(content.encode("utf-8"))
        files = {"file": ("report.txt", file_bytes, "text/plain")}
        response = requests.post(webhook_url, data={"content": "Report ist zu lang, angehängt als Datei."}, files=files, timeout=30)

    if response.status_code >= 400:
        print(f"Fehler beim Senden an Discord: {response.status_code} - {response.text}")
    else:
        print("Nachricht erfolgreich gesendet")

def main():
    parser = argparse.ArgumentParser(description="Sendet Textdatei an Discord Webhook.")
    parser.add_argument("--webhook", required=True, help="Discord Webhook URL")
    parser.add_argument("--file", required=True, help="Pfad zur Textdatei")
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        content = f.read()

    send_to_discord(args.webhook, content)

if __name__ == "__main__":
    main()
