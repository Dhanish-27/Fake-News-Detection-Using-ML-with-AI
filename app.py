"""
Flask Web Application for Fake News Detection
============================================

A web interface for the fake news detection system that allows users to:
- Input news articles for classification
- Choose between different ML models
- View confidence scores and detailed results
- See model performance metrics

Author: AI Assistant
Date: 2025
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'fake_news_detection_secret_key_2025'

class FakeNewsWebDetector:
    """
    Web version of the fake news detector for Flask application.
    """
    
    def __init__(self):
        """Initialize the web detector."""
        self.vectorizer = None
        self.models = {}
        self.model_info = {
            'Logistic Regression': {
                'description': 'Fast and interpretable model with high accuracy',
                'strengths': ['Fast training', 'Good baseline', 'Interpretable'],
                'accuracy': '~96%'
            },
            'Naive Bayes': {
                'description': 'Excellent for text classification with probabilistic approach',
                'strengths': ['Great for text', 'Handles high dimensions', 'Fast prediction'],
                'accuracy': '~93%'
            },
            'SVM': {
                'description': 'Powerful model that finds optimal decision boundaries',
                'strengths': ['High accuracy', 'Good generalization', 'Robust'],
                'accuracy': '~95%'
            }
        }
        
    def preprocess_text(self, text):
        """
        Preprocess text data by removing special characters, converting to lowercase,
        and removing extra whitespaces.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text) or not text:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_models(self):
        """Load pre-trained models and vectorizer."""
        try:
            # Load vectorizer
            if os.path.exists('fake_news_model_vectorizer.pkl'):
                with open('fake_news_model_vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
            else:
                raise FileNotFoundError("Vectorizer file not found. Please train models first.")
            
            # Load models
            model_files = {
                'Logistic Regression': 'fake_news_model_logistic_regression.pkl',
                'Naive Bayes': 'fake_news_model_naive_bayes.pkl',
                'SVM': 'fake_news_model_svm.pkl'
            }
            
            for name, filename in model_files.items():
                if os.path.exists(filename):
                    with open(filename, 'rb') as f:
                        self.models[name] = pickle.load(f)
                else:
                    print(f"Warning: Model file {filename} not found.")
            
            return len(self.models) > 0
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def predict(self, text, model_name='Logistic Regression'):
        """
        Predict whether a given text is fake or real news.
        
        Args:
            text (str): News text to classify
            model_name (str): Name of the model to use for prediction
            
        Returns:
            dict: Prediction results
        """
        if not self.vectorizer or model_name not in self.models:
            return {
                'error': f'Model "{model_name}" not available',
                'available_models': list(self.models.keys())
            }
        
        try:
            # Preprocess the input text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return {
                    'error': 'No valid text to process',
                    'prediction': None
                }
            
            # Transform using the trained vectorizer
            text_vector = self.vectorizer.transform([processed_text])
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(text_vector)[0]
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(text_vector)[0]
                confidence = {
                    'Fake News': float(probability[0]),
                    'Real News': float(probability[1])
                }
            else:
                confidence = {
                    'Fake News': 0.5,
                    'Real News': 0.5
                }
            
            result = {
                'text': text[:300] + "..." if len(text) > 300 else text,
                'prediction': 'Real News' if prediction == 1 else 'Fake News',
                'confidence': confidence,
                'model_used': model_name,
                'model_info': self.model_info.get(model_name, {}),
                'processed_text': processed_text[:200] + "..." if len(processed_text) > 200 else processed_text
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'prediction': None
            }

# Initialize the detector
detector = FakeNewsWebDetector()

@app.route('/')
def index():
    """Main page with news input form."""
    # Try to load models
    models_loaded = detector.load_models()
    
    return render_template('index.html', 
                         models_loaded=models_loaded,
                         available_models=list(detector.models.keys()),
                         model_info=detector.model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json() if request.is_json else request.form
        
        text = data.get('text', '').strip()
        model_name = data.get('model', 'Logistic Regression')
        
        if not text:
            return jsonify({
                'error': 'Please enter some text to analyze',
                'prediction': None
            })
        
        if len(text) < 10:
            return jsonify({
                'error': 'Text is too short. Please enter at least 10 characters.',
                'prediction': None
            })
        
        # Make prediction
        result = detector.predict(text, model_name)
        
        if 'error' in result:
            return jsonify(result)
        
        # Add additional analysis
        result['text_length'] = len(text)
        result['word_count'] = len(text.split())
        
        # Determine confidence level
        confidence_score = max(result['confidence'].values())
        if confidence_score >= 0.8:
            result['confidence_level'] = 'High'
        elif confidence_score >= 0.6:
            result['confidence_level'] = 'Medium'
        else:
            result['confidence_level'] = 'Low'
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'prediction': None
        })

@app.route('/train')
def train():
    """Page showing training status and instructions."""
    models_exist = (
        os.path.exists('fake_news_model_vectorizer.pkl') and
        os.path.exists('fake_news_model_logistic_regression.pkl') and
        os.path.exists('fake_news_model_naive_bayes.pkl') and
        os.path.exists('fake_news_model_svm.pkl')
    )
    
    return render_template('train.html', models_exist=models_exist)

@app.route('/about')
def about():
    """About page with project information."""
    return render_template('about.html', model_info=detector.model_info)

@app.route('/api/models')
def api_models():
    """API endpoint to get available models."""
    return jsonify({
        'models': list(detector.models.keys()),
        'model_info': detector.model_info
    })

@app.route('/api/stats')
def api_stats():
    """API endpoint to get dataset statistics."""
    try:
        # Try to get dataset info if files exist
        stats = {}
        
        if os.path.exists('NLP_Analysis/True.csv'):
            true_data = pd.read_csv('NLP_Analysis/True.csv')
            stats['true_news_count'] = len(true_data)
        
        if os.path.exists('NLP_Analysis/Fake.csv'):
            fake_data = pd.read_csv('NLP_Analysis/Fake.csv')
            stats['fake_news_count'] = len(fake_data)
        
        if stats:
            stats['total_articles'] = stats.get('true_news_count', 0) + stats.get('fake_news_count', 0)
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    if not os.path.exists('static'):
        os.makedirs('static')
    
    print("="*60)
    print("FAKE NEWS DETECTION WEB APPLICATION")
    print("="*60)
    print("Starting Flask application...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("="*60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)
