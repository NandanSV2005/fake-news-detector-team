# AI-Enhanced Fake News Detector

A hybrid fake news detection system combining traditional machine learning with generative AI (Grok/GPT-4/Gemini) for real-time fact-checking and verification.

## 🎯 Features

- **Hybrid Detection**: Combines ML pattern recognition with AI-powered fact-checking
- **Multi-AI Support**: Integrates Grok (xAI), GPT-4 (OpenAI), and Gemini (Google)
- **Real-time Verification**: Checks claims against current knowledge
- **Explainable Results**: Provides detailed explanations with source attribution
- **Current Events Ready**: Trained on 2025-2026 news including recent AI developments
- **Web Interface**: User-friendly Flask application

## 🏗️ Architecture

```
User Input → Preprocessing → ML Models (XGBoost, Random Forest, Logistic Regression)
                          ↓
                    Confidence Check
                          ↓
              Low Confidence? → AI Fact-Checker (Grok/GPT/Gemini)
                          ↓
                    Hybrid Scoring
                          ↓
            Detailed Report + Explanation + Sources
```

## 📋 Requirements

- Python 3.8+
- Virtual environment (recommended)
- API keys for at least one AI provider (optional but recommended)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd fake-news-detector
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys (Optional)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
# Get keys from:
# - Grok: https://console.x.ai
# - OpenAI: https://platform.openai.com
# - Gemini: https://ai.google.dev
```

### 4. Run the Pipeline

```bash
# Create enhanced dataset (150+ samples)
python enhanced_data_collection.py

# Preprocess text
python preprocessing.py

# Engineer features
python feature_engineering.py

# Train models
python model_training.py

# Start web application
python app.py
```

### 5. Open Browser

Navigate to: `http://localhost:5000`

## 🧪 Testing

### Test AI Fact-Checker

```bash
# Test with current events (2025-2026)
python test_current_events.py
```

### Test Individual Components

```bash
# Test AI fact-checker module
python ai_fact_checker.py

# Test data collection
python enhanced_data_collection.py
```

## 📊 Performance

### Expected Metrics

- **ML Model Accuracy**: 75-85% (on expanded dataset)
- **AI Verification Accuracy**: 90-95% (for clear cases)
- **Hybrid System Accuracy**: 85-95% (combined)
- **Response Time**: 3-8 seconds (ML + AI)

### Dataset

- **Total Samples**: 150+
- **True News**: ~50 articles (including 2025-2026 events)
- **Fake News**: ~85 articles (obvious + subtle misinformation)
- **Borderline**: ~15 articles (satire, opinion)

## 🔧 Configuration

Edit `config.py` or `.env` file to customize:

```python
# AI Provider (grok, openai, gemini)
DEFAULT_AI_PROVIDER=grok

# Hybrid scoring weights
ML_WEIGHT=0.4
AI_WEIGHT=0.6

# Confidence threshold to trigger AI
AI_CONFIDENCE_THRESHOLD=70
```

## 📁 Project Structure

```
fake-news-detector/
├── app.py                          # Flask web application
├── config.py                       # Configuration management
├── ai_fact_checker.py              # AI integration module
├── enhanced_data_collection.py     # Dataset creation
├── preprocessing.py                # Text preprocessing
├── feature_engineering.py          # Feature extraction
├── model_training.py               # ML model training
├── test_current_events.py          # Testing script
├── requirements.txt                # Dependencies
├── .env.example                    # Environment template
├── data/                           # Datasets
├── models/                         # Trained models
├── templates/                      # HTML templates
└── static/                         # CSS/JS assets
```

## 🤖 AI Providers

### Grok (xAI) - Recommended for Current Events

- **Strengths**: Real-time data, X integration, current events
- **Cost**: Pay-per-token
- **Setup**: Get API key from https://console.x.ai

### GPT-4 (OpenAI) - Most Reliable

- **Strengths**: Excellent reasoning, widely tested
- **Cost**: $0.03/1K tokens (input)
- **Setup**: Get API key from https://platform.openai.com

### Gemini (Google) - Free Tier Available

- **Strengths**: Free tier, fast, good for fact-checking
- **Cost**: Free tier + paid options
- **Setup**: Get API key from https://ai.google.dev

## ⚠️ Limitations

- **Not 100% Accurate**: AI can be wrong on edge cases
- **API Costs**: Grok/GPT-4 have per-token costs
- **Rate Limits**: Free tiers have usage restrictions
- **Latency**: AI calls add 2-4 seconds
- **Context Window**: Very long articles may be truncated
- **Bias**: Models have inherent biases from training data

## 💡 Best Use Cases

✅ **Excellent for:**
- Screening social media posts
- Quick verification of viral claims
- Educational tool for media literacy
- Identifying obvious fake news

⚠️ **Use with caution for:**
- Nuanced political analysis
- Satire vs. misinformation
- Context-dependent claims
- Breaking news (AI knowledge cutoff)

## 🔄 Future Enhancements

- [ ] Add more AI providers (Claude, Perplexity)
- [ ] Implement caching to reduce API costs
- [ ] Create browser extension
- [ ] Deploy to cloud (AWS/Heroku)
- [ ] Add fact-checking API integration
- [ ] Implement user feedback loop
- [ ] Add multilingual support

## 📝 License

MIT License - feel free to use and modify

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues or questions, please open an issue on GitHub.

---

**Note**: This is an educational/demonstration project. Always verify important information through multiple credible sources.
