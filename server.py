from flask import Flask
from threading import Thread
from pyngrok import ngrok
import os
from app import app  # Import your existing Flask app

def run_flask():
    app.run(host='0.0.0.0', port=5700)

def run_ngrok():
    try:
        # Kill any existing ngrok processes
        ngrok.kill()
        
        # Set up your auth token
        ngrok.set_auth_token("2ttqHuJ6l1brhg4OnZcAUksw0gY_2jgbnfoPfxHRrm74vwyRY")
        
        # Create tunnel
        public_url = ngrok.connect(5700).public_url
        print(f"\n * Public URL: {public_url}")
        
    except Exception as e:
        print(f"\n * Ngrok error: {str(e)}")
        print(" * Website is still accessible locally at http://localhost:5700")

if __name__ == '__main__':
    # Start Flask in a separate thread
    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    # Start Ngrok
    run_ngrok()
