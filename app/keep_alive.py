import time
import threading
import requests
import os

def ping_server():
    """
    Function to ping the server's own URL every 10 minutes to prevent Render from sleeping.
    """
    # The URL of the app on Render.
    url = "https://pdf-rag-chatbot-8mmj.onrender.com/"

    print(f"Keep-alive: Starting pinger for {url}")
    
    while True:
        try:
            # Ping every 10 minutes (600 seconds)
            time.sleep(600)
            response = requests.get(url)
            print(f"Keep-alive: Pinged {url} - Status Code: {response.status_code}")
        except Exception as e:
            print(f"Keep-alive: Error pinging server: {e}")

def start_keep_alive():
    """
    Starts the ping_server function in a separate background thread.
    """
    thread = threading.Thread(target=ping_server, daemon=True)
    thread.start()
    return thread
