from flask import Blueprint, request, jsonify, current_app
import os
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import io
import traceback
import sys
import base64
from eye_analysis_model import EyeAnalysisModel, InitializationError, PreprocessingError, AnalysisError

# Create Blueprint
eye_scan = Blueprint('eye_scan', __name__)

# Initialize model
model = None

def init_model():
    """Initialize the eye analysis model"""
    global model
    try:
        if model is None:
            print("\nInitializing eye analysis model...")
            model = EyeAnalysisModel()
            print("✓ Model initialized successfully")
        return True
    except Exception as e:
        print(f"\n❌ Error initializing model: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

def allowed_file(filename):
    """Check if file type is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def debug_log(message, error=None):
    """Helper function to log debug messages"""
    print(f"[DEBUG] {message}")
    if error:
        print(f"[ERROR] {str(error)}")
        print(f"[TRACEBACK]\n{traceback.format_exc()}")
        sys.stdout.flush()

@eye_scan.route('/api/analyze-eye', methods=['POST'])
def analyze_eye():
    """Handle eye analysis request"""
    try:
        # Initialize model if needed
        if not init_model():
            return jsonify({
                'error': 'Failed to initialize model',
                'status': 'Error',
                'status_detail': 'The eye analysis service failed to initialize',
                'status_color': 'red'
            }), 500

        # Validate request
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'status': 'Error',
                'status_detail': 'Please provide an image file',
                'status_color': 'red'
            }), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'error': 'No selected file',
                'status': 'Error',
                'status_detail': 'Please select an image file',
                'status_color': 'red'
            }), 400

        if not file or not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'status': 'Error',
                'status_detail': 'Please upload a valid image file (PNG, JPG, JPEG)',
                'status_color': 'red'
            }), 400

        # Read and process image
        try:
            # Read image file
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
                
            # Analyze image
            debug_log("Starting image analysis")
            result = model.analyze_image(image)
            debug_log("Analysis complete")
            
            # Generate visualization if condition detected
            if result.get('has_condition', False):
                try:
                    visualization = generate_visualization(image)
                    if visualization:
                        result['visualization'] = visualization
                except Exception as viz_error:
                    debug_log("Visualization generation failed", viz_error)
            
            return jsonify(result)
            
        except (InitializationError, PreprocessingError, AnalysisError) as e:
            debug_log("Analysis error", e)
            return jsonify({
                'error': str(e),
                'status': 'Error',
                'status_detail': 'Failed to analyze image',
                'status_color': 'red'
            }), 500
            
        except Exception as e:
            debug_log("Unexpected error", e)
            return jsonify({
                'error': 'Internal server error',
                'status': 'Error',
                'status_detail': 'An unexpected error occurred',
                'status_color': 'red'
            }), 500
            
    except Exception as e:
        debug_log("Route handler error", e)
        return jsonify({
            'error': 'Server error',
            'status': 'Error',
            'status_detail': 'The server encountered an error',
            'status_color': 'red'
        }), 500

def generate_visualization(image):
    """Generate heatmap visualization of the analysis"""
    try:
        debug_log("Starting visualization generation")
        
        # Convert image to numpy array if it's not already
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is in BGR format for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Create heatmap
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        # Blend original image with heatmap
        output = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', output)
        debug_log("Visualization generation completed successfully")
        return base64.b64encode(buffer).decode('utf-8')
        
    except Exception as e:
        debug_log("Error in generate_visualization", e)
        return None
