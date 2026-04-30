"""
Simple WebSocket test - just listen for any messages
"""

import json
import threading
import time
import websocket
import ssl
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

def on_message(ws, message):
    data = json.loads(message)
    print(f"RAW: {data}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("Connected! Authenticating...")
    auth_msg = {"action": "auth", "key": API_KEY, "secret": SECRET_KEY}
    ws.send(json.dumps(auth_msg))
    
    # Subscribe to symbols
    subscribe_msg = {
        "action": "subscribe",
        "trades": ["NVDA", "AAPL"],
        "quotes": ["NVDA", "AAPL"]
    }
    time.sleep(1)  # Wait for auth to complete
    ws.send(json.dumps(subscribe_msg))
    print("Subscribed to NVDA and AAPL")

print("Testing WebSocket...")
ws_url = "wss://stream.data.alpaca.markets/v2/iex"

ws = websocket.WebSocketApp(
    ws_url,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)

wst = threading.Thread(target=ws.run_forever, kwargs={'sslopt': {"cert_reqs": ssl.CERT_NONE}})
wst.daemon = True
wst.start()

print("Listening for 30 seconds... (will show RAW messages)\n")

try:
    time.sleep(30)
except KeyboardInterrupt:
    pass
finally:
    ws.close()
    print("\nDone!")
