# preprocessing.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np

class TextPreprocessor:
    def __init__(self):
        print("Initializing Text Preprocessor...")
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-eng', quiet=True)
        except:
            print("NLTK downloads completed")
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        print("Text Preprocessor ready!")
    
    def clean_text(self, text):
        """Clean and preprocess a single text string"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user @ references and '#' from hashtags
        text = re.sub(r'\@\w+|\#', '', text)
        
        # Remove special characters and numbers (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize text and lemmatize words"""
        if not text:
            return ""
        
        try:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and short words, then lemmatize
            processed_tokens = []
            for word in tokens:
                if word not in self.stop_words and len(word) > 2:
                    lemmatized_word = self.lemmatizer.lemmatize(word)
                    processed_tokens.append(lemmatized_word)
            
            return ' '.join(processed_tokens)
        except:
            # Fallback: simple split if word_tokenize fails
            tokens = text.split()
            processed_tokens = []
            for word in tokens:
                if word not in self.stop_words and len(word) > 2:
                    processed_tokens.append(word)
            
            return ' '.join(processed_tokens)
    
    def extract_basic_features(self, text):
        """Extract basic text features for analysis"""
        if not text:
            return {
                'word_count': 0,
                'char_count': 0,
                'avg_word_length': 0,
                'sentence_count': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'uppercase_ratio': 0,
                'stopword_count': 0
            }
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Count uppercase letters in original text
        uppercase_count = sum(1 for c in text if c.isupper())
        
        features = {
            'word_count': len(words),
            'char_count': len(text),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'sentence_count': len(sentences),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': uppercase_count / len(text) if len(text) > 0 else 0,
            'stopword_count': sum(1 for w in words if w.lower() in self.stop_words)
        }
        
        return features
    
    def process_dataframe(self, df, text_column='text'):
        """Process entire dataframe column"""
        print(f"Processing dataframe with {len(df)} rows...")
        
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Clean text
        print("  - Cleaning text...")
        df_processed['cleaned_text'] = df_processed[text_column].apply(self.clean_text)
        
        # Tokenize and lemmatize
        print("  - Tokenizing and lemmatizing...")
        df_processed['processed_text'] = df_processed['cleaned_text'].apply(self.tokenize_and_lemmatize)
        
        # Extract features
        print("  - Extracting features...")
        features_list = []
        for text in df_processed['cleaned_text']:
            features_list.append(self.extract_basic_features(text))
        
        # Add features to dataframe
        features_df = pd.DataFrame(features_list)
        df_processed = pd.concat([df_processed, features_df], axis=1)
        
        print(f"Processing complete! Added {len(features_df.columns)} new features.")
        
        return df_processed

def test_preprocessor():
    """Test function for the preprocessor"""
    print("=== Testing Text Preprocessor ===\n")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Test with a sample news text
    sample_text = "BREAKING: Scientists discovered NEW species in Amazon rainforest! Check: https://news.com #Science"
    
    # Show sample processing
    print("\n" + "="*60)
    print("SAMPLE TEXT PROCESSING DEMO")
    print("="*60)
    
    print(f"\nOriginal Text:\n{sample_text}")
    
    cleaned = preprocessor.clean_text(sample_text)
    print(f"\nAfter Cleaning:\n{cleaned}")
    
    processed = preprocessor.tokenize_and_lemmatize(cleaned)
    print(f"\nAfter Tokenization & Lemmatization:\n{processed}")
    
    # Test with our dataset
    print("\n" + "="*60)
    print("PROCESSING SAMPLE DATASET")
    print("="*60)
    
    try:
        # Load the data we created earlier
        df = pd.read_csv('data/news_data.csv')
        print(f"\nLoaded dataset with {len(df)} articles")
        
        # Process the dataframe
        df_processed = preprocessor.process_dataframe(df)
        
        # Show results
        print("\nProcessed Data Sample:")
        print(df_processed[['text', 'cleaned_text', 'word_count', 'label']].head())
        
        # Save processed data
        output_path = 'data/processed_news_data.csv'
        df_processed.to_csv(output_path, index=False)
        print(f"\nSaved processed data to: {output_path}")
        
        # Show some statistics
        print("\nDataset Statistics:")
        print(f"Average word count: {df_processed['word_count'].mean():.1f}")
        print(f"Average character count: {df_processed['char_count'].mean():.1f}")
        print(f"Fake news avg word count: {df_processed[df_processed['label']==0]['word_count'].mean():.1f}")
        print(f"True news avg word count: {df_processed[df_processed['label']==1]['word_count'].mean():.1f}")
        
    except FileNotFoundError:
        print("Data file not found. Run data_collection.py first!")
    
    print("\n=== Preprocessor Test Complete ===")

if __name__ == "__main__":
    test_preprocessor()