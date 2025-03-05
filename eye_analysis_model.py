import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import traceback
import cv2

class EyeConditionModel(nn.Module):
    def __init__(self):
        super(EyeConditionModel, self).__init__()
        # Define model architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
        # Separate heads for different tasks
        self.condition_head = nn.Linear(256, 3)  # 3 conditions
        self.severity_head = nn.Linear(256, 3)   # 3 severity levels
        
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Shared classifier
        x = self.classifier(x)
        
        # Task-specific heads
        condition = self.condition_head(x)
        severity = self.severity_head(x)
        
        return condition, severity

class EyeAnalysisModel:
    def __init__(self):
        self.detection_model = None
        self.classification_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.initialized = False
        self.root_dir = os.path.dirname(__file__)
        self.load_models()
        
    def load_models(self):
        """Load both detection and classification models"""
        try:
            # Load detection model (TensorFlow)
            detection_path = os.path.join(self.root_dir, 'model_after_testing.keras')
            print(f"Loading detection model from {detection_path}")
            if os.path.exists(detection_path):
                self.detection_model = load_model(detection_path)
                print("Detection model loaded successfully")
            else:
                raise FileNotFoundError(f"Detection model not found at {detection_path}")
            
            # Load classification model (PyTorch)
            classification_path = os.path.join(self.root_dir, 'model_epoch26_acc94.76.pt')
            print(f"Loading classification model from {classification_path}")
            if os.path.exists(classification_path):
                self.classification_model = EyeConditionModel().to(self.device)
                
                # Load the state dict
                state_dict = torch.load(classification_path, map_location=self.device)
                print(f"State dict keys: {state_dict.keys() if isinstance(state_dict, dict) else 'Not a dict'}")
                
                if isinstance(state_dict, dict):
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    # Remove module prefix if present
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    self.classification_model.load_state_dict(state_dict, strict=False)
                    print("Classification model weights loaded")
                else:
                    raise ValueError("Invalid state dict format")
                
                self.classification_model.eval()
                print("Classification model ready")
            else:
                raise FileNotFoundError(f"Classification model not found at {classification_path}")
            
            self.initialized = True
            print("Both models initialized and ready")
            
        except Exception as e:
            print(f"\nError loading models: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            raise InitializationError(f"Failed to initialize models: {str(e)}")

    def preprocess_image_tf(self, image):
        """Preprocess image for TensorFlow model"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to expected input size
            image = cv2.resize(image, (224, 224))
            
            # Normalize to [0,1]
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image for TF: {str(e)}")
            raise PreprocessingError(f"Failed to preprocess image for detection: {str(e)}")

    def preprocess_image_torch(self, image):
        """Preprocess image for PyTorch model"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize and convert to tensor
            image = cv2.resize(image, (224, 224))
            image = torch.from_numpy(image.transpose((2, 0, 1))).float()
            
            # Normalize
            image = image / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
            
            # Add batch dimension
            image = image.unsqueeze(0).to(self.device)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image for PyTorch: {str(e)}")
            raise PreprocessingError(f"Failed to preprocess image for classification: {str(e)}")

    def analyze_image(self, image):
        """Analyze eye image using both models"""
        try:
            if not self.initialized:
                raise InitializationError("Models not properly initialized")
                
            print("\n=== Starting Image Analysis ===")
            print(f"Input image shape: {image.shape}")
            
            # First use detection model to check for eye condition
            tf_input = self.preprocess_image_tf(image)
            print(f"Detection model input shape: {tf_input.shape}")
            
            # Get detection prediction and handle different output formats
            detection_pred = self.detection_model.predict(tf_input, verbose=0)
            print(f"Raw detection output shape: {detection_pred.shape}")
            print(f"Raw detection output: {detection_pred}")
            
            # Handle different possible output shapes and formats
            if isinstance(detection_pred, list):
                detection_pred = detection_pred[0]  # Take first output if multiple outputs
            
            # Convert to numpy array if needed
            if not isinstance(detection_pred, np.ndarray):
                detection_pred = np.array(detection_pred)
            
            # Reshape if needed
            if len(detection_pred.shape) == 1:
                # Single value output - interpret as condition probability
                condition_prob = float(detection_pred[0])
                healthy_prob = 1.0 - condition_prob
            elif len(detection_pred.shape) == 2:
                # (batch, classes) output
                detection_pred = detection_pred[0]  # Take first batch
                if len(detection_pred) == 1:
                    # Single class output - interpret as condition probability
                    condition_prob = float(detection_pred[0])
                    healthy_prob = 1.0 - condition_prob
                else:
                    # Multiple class output
                    healthy_prob = float(detection_pred[0])
                    condition_prob = float(detection_pred[1])
            elif len(detection_pred.shape) == 3:
                # (batch, timesteps, classes) output
                detection_pred = detection_pred[0, -1]  # Take last timestep of first batch
                if len(detection_pred) == 1:
                    # Single class output
                    condition_prob = float(detection_pred[0])
                    healthy_prob = 1.0 - condition_prob
                else:
                    # Multiple class output
                    healthy_prob = float(detection_pred[0])
                    condition_prob = float(detection_pred[1])
            else:
                raise ValueError(f"Unexpected detection model output shape: {detection_pred.shape}")
            
            print(f"\nProcessed Detection Results:")
            print(f"- Healthy probability: {healthy_prob:.3f}")
            print(f"- Condition probability: {condition_prob:.3f}")
            
            # Consider healthy if healthy probability is higher and above threshold
            HEALTHY_THRESHOLD = 0.7  # Increased threshold for more accurate detection
            CONDITION_THRESHOLD = 0.6  # Threshold for condition detection
            
            is_healthy = healthy_prob > HEALTHY_THRESHOLD and healthy_prob > condition_prob
            has_condition = condition_prob > CONDITION_THRESHOLD
            
            print(f"\nClassification Decision:")
            print(f"- Is Healthy: {is_healthy}")
            print(f"- Has Condition: {has_condition}")
            
            # Return healthy result if classified as healthy
            if is_healthy and not has_condition:
                print("\nClassified as HEALTHY")
                return {
                    'has_condition': False,
                    'condition': 'Normal',
                    'severity': 'None',
                    'confidence': healthy_prob,
                    'severity_confidence': 1.0,
                    'status': 'Healthy Eye',
                    'status_detail': 'Healthy Eye',
                    'status_color': 'green',
                    'severity_description': 'Your eyes appear healthy with no signs of infection or inflammation.',
                    'recommendations': [
                        'Continue regular eye check-ups',
                        'Maintain good eye hygiene',
                        'Use protective eyewear when needed',
                        'Take regular breaks from screen time'
                    ]
                }
            
            # If condition detected, use classification model
            print("\nAnalyzing condition type and severity...")
            torch_input = self.preprocess_image_torch(image)
            print(f"Classification model input shape: {torch_input.shape}")
            
            with torch.no_grad():
                try:
                    condition_output, severity_output = self.classification_model(torch_input)
                    print(f"\nModel raw outputs:")
                    print(f"- Condition output shape: {condition_output.shape}")
                    print(f"- Severity output shape: {severity_output.shape}")
                    
                    # Get probabilities
                    condition_probs = torch.softmax(condition_output, dim=1)[0]
                    severity_probs = torch.softmax(severity_output, dim=1)[0]
                    
                    # Get predictions
                    condition_idx = condition_probs.argmax().item()
                    severity_idx = severity_probs.argmax().item()
                    
                    # Map to labels
                    conditions = ['Bacterial Conjunctivitis', 'Viral Conjunctivitis', 'Allergic Conjunctivitis']
                    severity_levels = ['Mild', 'Moderate', 'Severe']
                    
                    condition = conditions[condition_idx]
                    severity = severity_levels[severity_idx]
                    confidence = float(condition_probs[condition_idx])
                    severity_confidence = float(severity_probs[severity_idx])
                    
                    print(f"\nCondition Analysis Results:")
                    print(f"- Detected Condition: {condition}")
                    print(f"- Severity Level: {severity}")
                    print(f"- Confidence: {confidence:.3f}")
                    print(f"- Severity Confidence: {severity_confidence:.3f}")
                    
                    return {
                        'has_condition': True,
                        'condition': condition,
                        'severity': severity,
                        'confidence': confidence,
                        'severity_confidence': severity_confidence,
                        'status': 'Eye Flu Detected',
                        'status_detail': 'Eye Flu Detected',
                        'status_color': 'red',
                        'severity_description': self._get_severity_description(condition, severity),
                        'recommendations': self._get_recommendations(condition, severity)
                    }
                    
                except Exception as e:
                    print(f"Error during model forward pass: {str(e)}")
                    print(f"Traceback: {traceback.format_exc()}")
                    raise AnalysisError(f"Failed to analyze condition: {str(e)}")
            
        except Exception as e:
            print(f"Error in analyze_image: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise AnalysisError(f"Failed to analyze image: {str(e)}")

    def _get_severity_description(self, condition, severity):
        """Get detailed description based on condition and severity"""
        if not condition or condition == 'Normal':
            return "No signs of eye infection or inflammation detected."
        
        condition_desc = {
            'Bacterial Conjunctivitis': 'Bacterial eye infection causing redness and discharge.',
            'Viral Conjunctivitis': 'Viral infection with eye redness and watery discharge.',
            'Allergic Conjunctivitis': 'Allergic reaction causing eye irritation and itching.',
        }
        
        severity_desc = {
            'Mild': 'Early stage with minimal symptoms.',
            'Moderate': 'Developed symptoms requiring attention.',
            'Severe': 'Advanced symptoms needing immediate care.'
        }
        
        base_desc = condition_desc.get(condition, 'Eye condition detected.')
        severity_info = severity_desc.get(severity, '')
        
        return f"{base_desc} {severity_info}"

    def _get_recommendations(self, condition, severity):
        """Get recommendations based on condition and severity"""
        if not condition or condition == 'Normal':
            return [
                'Continue regular eye check-ups',
                'Maintain good eye hygiene',
                'Use protective eyewear when needed',
                'Take regular breaks from screen time'
            ]
        
        base_recommendations = [
            'Schedule an appointment with an eye specialist',
            'Avoid touching or rubbing your eyes',
            'Maintain good eye hygiene'
        ]
        
        condition_specific = {
            'Bacterial Conjunctivitis': [
                'Use prescribed antibiotic eye drops',
                'Clean eyelids with warm water',
                'Change pillowcase daily'
            ],
            'Viral Conjunctivitis': [
                'Apply cold compresses',
                'Use artificial tears',
                'Avoid contact with others'
            ],
            'Allergic Conjunctivitis': [
                'Avoid known allergens',
                'Use antihistamine eye drops',
                'Apply cool compresses'
            ]
        }
        
        severity_specific = {
            'Mild': ['Monitor symptoms for any changes'],
            'Moderate': [
                'Take time off from work/school',
                'Avoid wearing contact lenses'
            ],
            'Severe': [
                'Seek immediate medical attention',
                'Avoid bright lights',
                'Rest eyes frequently'
            ]
        }
        
        recommendations = base_recommendations.copy()
        
        if condition in condition_specific:
            recommendations.extend(condition_specific[condition])
        
        if severity in severity_specific:
            recommendations.extend(severity_specific[severity])
        
        return recommendations

class InitializationError(Exception):
    pass

class PreprocessingError(Exception):
    pass

class AnalysisError(Exception):
    pass
