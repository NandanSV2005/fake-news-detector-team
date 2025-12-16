# config.py - Centralized configuration for AI-Enhanced Fake News Detector
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# AI Provider Configuration
# ============================================

AI_CONFIG = {
    'grok': {
        'api_key': os.getenv('GROK_API_KEY', ''),
        'endpoint': 'https://api.x.ai/v1/chat/completions',
        'model': 'grok-beta',
        'enabled': bool(os.getenv('GROK_API_KEY')),
        'max_tokens': 2000,
        'temperature': 0.3  # Lower for more factual responses
    },
    'openai': {
        'api_key': os.getenv('OPENAI_API_KEY', ''),
        'model': 'gpt-4-turbo-preview',
        'enabled': bool(os.getenv('OPENAI_API_KEY')),
        'max_tokens': 2000,
        'temperature': 0.3
    },
    'gemini': {
        'api_key': os.getenv('GEMINI_API_KEY', ''),
        'model': 'gemini-pro',
        'enabled': bool(os.getenv('GEMINI_API_KEY')),
        'max_tokens': 2000,
        'temperature': 0.3
    }
}

# Default provider (fallback order: grok -> openai -> gemini)
DEFAULT_AI_PROVIDER = os.getenv('DEFAULT_AI_PROVIDER', 'grok')

# ============================================
# Hybrid Model Configuration
# ============================================

HYBRID_CONFIG = {
    # Weights for combining ML and AI predictions
    'ml_weight': float(os.getenv('ML_WEIGHT', '0.4')),
    'ai_weight': float(os.getenv('AI_WEIGHT', '0.6')),
    
    # Confidence threshold to trigger AI verification
    # If ML confidence < threshold, use AI for verification
    'confidence_threshold': int(os.getenv('AI_CONFIDENCE_THRESHOLD', '70')),
    
    # Always use AI (even if ML is confident)
    'use_ai_for_all': os.getenv('USE_AI_VERIFICATION', 'true').lower() == 'true',
    
    # Maximum time to wait for AI response (seconds)
    'ai_timeout': 30,
    
    # Enable caching to reduce API costs
    'enable_cache': True,
    'cache_ttl': 3600  # 1 hour
}

# ============================================
# Feature Engineering Configuration
# ============================================

FEATURE_CONFIG = {
    'max_tfidf_features': 1500,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95
}

# ============================================
# Model Paths
# ============================================

MODEL_PATHS = {
    'ensemble': 'models/ensemble_model.pkl',
    'tfidf': 'models/tfidf_vectorizer.pkl',
    'scaler': 'models/scaler.pkl',
    'best_model': 'models/best_model.pkl'
}

# ============================================
# Sensational & Reliable Word Lists
# ============================================

SENSATIONAL_WORDS = [
    'breaking', 'urgent', 'shocking', 'secret', 'exposed', 'revealed',
    'leaked', 'whistleblower', 'truth', 'hidden', 'suppressed', 'wake up',
    'conspiracy', 'cover-up', 'alert', 'emergency', 'warning', 'danger',
    'scandal', 'exclusive', 'bombshell', 'explosive', 'stunning'
]

RELIABLE_WORDS = [
    'study', 'research', 'according to', 'published', 'journal',
    'university', 'scientists', 'researchers', 'data', 'analysis',
    'report', 'official', 'ministry', 'department', 'confirmed',
    'experts', 'findings', 'evidence', 'survey', 'results', 'peer-reviewed'
]

# ============================================
# Fact-Checking Prompt Template
# ============================================

FACT_CHECK_PROMPT = """You are an expert fact-checker analyzing news content for authenticity.

Analyze this news text:
\"\"\"{text}\"\"\"

Provide a comprehensive analysis in JSON format with these fields:

1. "claims": List of verifiable factual claims (array of strings)
2. "verification": For each claim, verify against current knowledge
   - claim: the statement
   - verdict: "TRUE", "FALSE", "UNVERIFIED", or "MISLEADING"
   - confidence: 0-100
   - reasoning: brief explanation
3. "source_credibility": Assessment of writing style and credibility markers
   - sensationalism_score: 0-100 (higher = more sensational)
   - reliability_indicators: list of credible elements found
   - red_flags: list of concerning elements
4. "overall_assessment":
   - prediction: "FAKE" or "TRUE"
   - confidence: 0-100
   - explanation: detailed reasoning (2-3 sentences)
5. "sources": Relevant credible sources to verify claims (if applicable)

Be objective and base your analysis on factual accuracy, not political bias.
"""

# ============================================
# Logging Configuration
# ============================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'logs/fake_news_detector.log'
}

# ============================================
# Helper Functions
# ============================================

def get_available_providers():
    """Return list of AI providers with valid API keys"""
    return [name for name, config in AI_CONFIG.items() if config['enabled']]

def get_provider_config(provider_name):
    """Get configuration for specific AI provider"""
    return AI_CONFIG.get(provider_name, {})

def validate_config():
    """Validate configuration and return warnings"""
    warnings = []
    
    # Check if at least one AI provider is configured
    available = get_available_providers()
    if not available:
        warnings.append("No AI providers configured. Set API keys in .env file.")
    
    # Check weights sum to 1.0
    total_weight = HYBRID_CONFIG['ml_weight'] + HYBRID_CONFIG['ai_weight']
    if abs(total_weight - 1.0) > 0.01:
        warnings.append(f"ML and AI weights should sum to 1.0 (current: {total_weight})")
    
    return warnings

# Run validation on import
if __name__ != '__main__':
    config_warnings = validate_config()
    if config_warnings:
        print("⚠️  Configuration Warnings:")
        for warning in config_warnings:
            print(f"  - {warning}")
