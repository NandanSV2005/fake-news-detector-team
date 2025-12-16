# app.py - Complete working version
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import os

app = Flask(__name__)

# Ensure models exist
MODEL_PATH = 'models/ensemble_model.pkl'
TFIDF_PATH = 'models/tfidf_vectorizer.pkl'
SCALER_PATH = 'models/scaler.pkl'

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

# Load models with error handling
def load_models():
    models = {}
    
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                models['classifier'] = pickle.load(f)
        else:
            print(f"Warning: {MODEL_PATH} not found")
            models['classifier'] = None
            
        if os.path.exists(TFIDF_PATH):
            with open(TFIDF_PATH, 'rb') as f:
                models['vectorizer'] = pickle.load(f)
        else:
            print(f"Warning: {TFIDF_PATH} not found")
            models['vectorizer'] = None
            
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                models['scaler'] = pickle.load(f)
        else:
            print(f"Warning: {SCALER_PATH} not found")
            models['scaler'] = None
            
    except Exception as e:
        print(f"Error loading models: {e}")
        models = {
            'classifier': None,
            'vectorizer': None,
            'scaler': None
        }
    
    return models

# Initialize models
models = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': models['classifier'] is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        # Check if models are loaded
        if not models['classifier'] or not models['vectorizer']:
            return jsonify({
                'error': 'Models not loaded. Please run model training first.',
                'prediction': 'ERROR',
                'confidence': 0,
                'probability': {'FAKE': 0.5, 'TRUE': 0.5}
            }), 500
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Extract features
        if models['vectorizer']:
            tfidf_features = models['vectorizer'].transform([processed_text]).toarray()
        else:
            tfidf_features = np.zeros((1, 5000))  # Default dimensions
        
        # Enhanced features matching feature_engineering.py (20 features)
        features_list = []
        
        sensational_words = ['breaking', 'urgent', 'shocking', 'secret', 'exposed', 
                           'revealed', 'leaked', 'whistleblower', 'truth', 'hidden',
                           'suppressed', 'wake up', 'conspiracy', 'cover-up', 'alert',
                           'emergency', 'warning', 'danger', 'scandal', 'exclusive']
        
        reliable_words = ['study', 'research', 'according to', 'published', 'journal',
                         'university', 'scientists', 'researchers', 'data', 'analysis',
                         'report', 'official', 'ministry', 'department', 'confirmed',
                         'experts', 'findings', 'evidence', 'survey', 'results']
        
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            additional_features = np.zeros((1, 20))
        else:
            # Sensationalism indicators
            sensational_count = sum(1 for word in words if any(sw in word for sw in sensational_words))
            
            # Reliability indicators
            reliable_count = sum(1 for word in words if any(rw in word for rw in reliable_words))
            
            # Exaggeration markers
            exaggeration_words = ['completely', 'absolutely', 'totally', 'entirely', 
                                'perfectly', 'extremely', 'incredibly', 'unbelievably']
            exaggeration_count = sum(1 for word in words if word in exaggeration_words)
            
            # Miracle cure claims
            miracle_words = ['miracle cure', 'instant cure', 'cures instantly', 'simple trick',
                           'one simple', 'doctors hate', 'they don\'t want you to know',
                           'big pharma', 'hidden cure', 'secret remedy']
            miracle_count = 0
            for mw in miracle_words:
                if mw in text_lower:
                    miracle_count += 1
            
            # Text statistics
            word_count = len(words)
            char_count = len(text)
            exclamation_count = text.count('!')
            question_count = text.count('?')
            all_caps_count = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
            
            # Calculate ratios
            sensational_ratio = sensational_count / word_count if word_count > 0 else 0
            reliable_ratio = reliable_count / word_count if word_count > 0 else 0
            exclamation_ratio = exclamation_count / (word_count / 100) if word_count > 0 else 0
            caps_ratio = all_caps_count / (word_count / 100) if word_count > 0 else 0
            
            # Length features
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            
            # Dollar sign count (financial sensationalism)
            dollar_count = text.count('$')
            
            # Percentage count (exaggerated claims)
            percent_count = text.count('%')
            
            # Multiple exclamation flag
            multi_exclamation = 1 if exclamation_count > 2 else 0
            
            # ALL CAPS flag
            all_caps_flag = 1 if caps_ratio > 10 else 0
            
            feature_vector = [
                word_count,
                char_count,
                sensational_count,
                reliable_count,
                exaggeration_count,
                miracle_count,
                exclamation_count,
                question_count,
                all_caps_count,
                dollar_count,
                percent_count,
                sensational_ratio,
                reliable_ratio,
                exclamation_ratio,
                caps_ratio,
                avg_word_length,
                multi_exclamation,
                all_caps_flag,
                1 if sensational_count > reliable_count else 0,  # More sensational than reliable
                1 if exclamation_count > 3 else 0  # Excessive exclamations
            ]
            
            additional_features = np.array([feature_vector])
            
        if models['scaler']:
            additional_features = models['scaler'].transform(additional_features)
        
        # Combine all features
        if tfidf_features.shape[1] > 0 and additional_features.shape[1] > 0:
            features = np.concatenate([tfidf_features, additional_features], axis=1)
        else:
            features = additional_features if tfidf_features.shape[1] == 0 else tfidf_features
        
        # Make prediction
        prediction = models['classifier'].predict(features)[0]
        probabilities = models['classifier'].predict_proba(features)[0]
        
        # Map prediction to label (1 = TRUE, 0 = FAKE)
        label = 'TRUE' if prediction == 1 else 'FAKE'
        confidence = max(probabilities) # Keep as 0-1 float for frontend
        
        # Generate key indicators based on features
        key_indicators = []
        if sensational_count > 0:
            key_indicators.append(f"{sensational_count} sensational words detected")
        if reliable_count > 0:
            key_indicators.append(f"{reliable_count} reliability indicators found")
        if exaggeration_count > 0:
            key_indicators.append("Exaggerated language detected")
        if miracle_count > 0:
            key_indicators.append("Miracle cure claims found")
        if all_caps_flag:
            key_indicators.append("Excessive use of ALL CAPS")
        if multi_exclamation:
            key_indicators.append("Excessive exclamation marks")
            
        # Prepare response matching frontend expectations
        response = {
            'is_fake': bool(prediction == 0),
            'label': label,
            'confidence': float(confidence),
            'true_probability': float(probabilities[1]),
            'fake_probability': float(probabilities[0]),
            'sensational_score': int(sensational_count),
            'reliable_score': int(reliable_count),
            'key_indicators': key_indicators,
            'text': text,
            'prediction_mode': 'ML-Only'
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'error': str(e),
            'prediction': 'ERROR',
            'confidence': 0,
            'probability': {'FAKE': 0.5, 'TRUE': 0.5}
        }), 500

if __name__ == '__main__':
    print("Starting Fake News Detector Server...")
    print(f"Model loaded: {models['classifier'] is not None}")
    print(f"Vectorizer loaded: {models['vectorizer'] is not None}")
    print(f"Scaler loaded: {models['scaler'] is not None}")
    print("Server running at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)