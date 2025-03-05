from flask import Flask, render_template
import os
import webbrowser
from threading import Timer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def open_browser():
    webbrowser.open_new('http://localhost:5700/')

if __name__ == '__main__':
    print("Starting EyeAI Diagnostic Platform...")
    Timer(1.5, open_browser).start()
    port = 5700
    print(f"Server starting on http://localhost:{port}")
    print("The application will open in your default browser automatically.")
    print("Press Ctrl+C to stop the server.")
    app.run(host='0.0.0.0', port=port, debug=False)
