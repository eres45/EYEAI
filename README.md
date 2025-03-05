# Eye Flu Classification System

An AI-powered web application for detecting and classifying eye flu conditions using deep learning.

## Features

- AI-powered eye flu detection
- Real-time image analysis
- Detailed medical recommendations
- Comprehensive knowledge base
- Privacy-focused design
- HIPAA and GDPR compliant

## Project Structure

```
eye-flu/
├── static/             # Static files (CSS, JS)
├── templates/          # HTML templates
├── uploads/            # Temporary image upload directory
├── app.py             # Flask application
├── requirements.txt   # Python dependencies
└── model_after_testing.keras  # Trained AI model
```

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Unix/MacOS:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure your virtual environment is activated

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:4000
   ```

## Usage

1. Click on "Upload Image & Analyze" on the homepage
2. Upload an eye image or use the camera to take a photo
3. Wait for the AI analysis to complete
4. View the results and recommendations
5. Download or share the report if needed

## Security and Privacy

- All images are processed securely and not stored permanently
- Data transmission is encrypted
- Compliant with HIPAA and GDPR regulations
- Regular security audits and updates

## Technologies Used

- Frontend: HTML5, CSS3, JavaScript
- Backend: Python, Flask
- AI/ML: TensorFlow, Keras
- Security: Flask-CORS, Werkzeug

## Support

For support, email support@eyehealthai.com or visit our Knowledge Base.

## License

Copyright © 2024 Eye Flu Classification System. All rights reserved.
