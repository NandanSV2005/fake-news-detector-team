# feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import os
import re

class FeatureEngineer:
    def __init__(self, max_features=500):
        """
        Initialize feature engineer
        max_features: Maximum number of features to create
        """
        print(f"Initializing Feature Engineer with max_features={max_features}")
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        
    def create_tfidf_features(self, texts, fit=True):
        """
        Create TF-IDF features from text
        TF-IDF = Term Frequency-Inverse Document Frequency
        Higher weight for words that are important in a document but rare in the collection
        """
        print("Creating TF-IDF features...")
        
        if self.tfidf_vectorizer is None or fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),  # Use single words and pairs of words
                stop_words='english'
            )
        
        if fit:
            features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            features = self.tfidf_vectorizer.transform(texts)
        
        print(f"TF-IDF Features shape: {features.shape}")
        return features
    
    def create_enhanced_features(self, texts):
        """
        Create enhanced linguistic and stylistic features
        """
        print("Creating enhanced features...")
        
        features_list = []
        
        sensational_words = ['breaking', 'urgent', 'shocking', 'secret', 'exposed', 
                           'revealed', 'leaked', 'whistleblower', 'truth', 'hidden',
                           'suppressed', 'wake up', 'conspiracy', 'cover-up', 'alert',
                           'emergency', 'warning', 'danger', 'scandal', 'exclusive']
        
        reliable_words = ['study', 'research', 'according to', 'published', 'journal',
                         'university', 'scientists', 'researchers', 'data', 'analysis',
                         'report', 'official', 'ministry', 'department', 'confirmed',
                         'experts', 'findings', 'evidence', 'survey', 'results']
        
        for text in texts:
            text_lower = text.lower()
            words = text_lower.split()
            
            if not words:
                features_list.append([0] * 20)
                continue
            
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
            
            features_list.append(feature_vector)
        
        features_array = np.array(features_list)
        print(f"Enhanced Features shape: {features_array.shape}")
        return features_array
    
    def combine_features(self, texts, use_tfidf=True, use_enhanced=True):
        """
        Combine multiple feature types
        """
        print("\nCombining features...")
        
        feature_list = []
        
        if use_tfidf:
            tfidf_features = self.create_tfidf_features(texts, fit=True)
            feature_list.append(tfidf_features)
            print(f"  Added TF-IDF features ({tfidf_features.shape[1]} dimensions)")
        
        if use_enhanced:
            enhanced_features = self.create_enhanced_features(texts)
            # Convert to sparse matrix
            from scipy import sparse
            enhanced_sparse = sparse.csr_matrix(enhanced_features)
            feature_list.append(enhanced_sparse)
            print(f"  Added Enhanced features ({enhanced_features.shape[1]} dimensions)")
        
        # Combine all features
        if len(feature_list) > 1:
            from scipy.sparse import hstack
            combined_features = hstack(feature_list)
        else:
            combined_features = feature_list[0]
        
        print(f"\nFinal combined features shape: {combined_features.shape}")
        return combined_features
    
    def save_vectorizers(self, save_dir='models'):
        """
        Save vectorizers for later use
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if self.tfidf_vectorizer:
            with open(os.path.join(save_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            print(f"Saved TF-IDF vectorizer to {save_dir}/tfidf_vectorizer.pkl")
    
    def load_vectorizers(self, save_dir='models'):
        """
        Load saved vectorizers
        """
        tfidf_path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')
        
        if os.path.exists(tfidf_path):
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            print(f"Loaded TF-IDF vectorizer from {tfidf_path}")

def test_feature_engineering():
    """
    Test the feature engineering module
    """
    print("="*60)
    print("TESTING FEATURE ENGINEERING MODULE")
    print("="*60)
    
    try:
        # Load processed data
        df = pd.read_csv('data/processed_news_data.csv')
        print(f"\nLoaded {len(df)} processed articles")
        
        # Initialize feature engineer
        engineer = FeatureEngineer(max_features=300)
        
        # Get processed texts
        texts = df['processed_text'].fillna('').tolist()
        labels = df['label'].values
        
        print(f"\nSample texts for feature engineering:")
        for i in range(min(2, len(texts))):
            print(f"  Text {i+1}: {texts[i][:80]}...")
        
        # Create combined features
        print("\n" + "-"*40)
        X = engineer.combine_features(
            texts, 
            use_tfidf=True, 
            use_enhanced=True
        )
        
        # Show feature information
        print(f"\nFeature Matrix Information:")
        print(f"  Shape: {X.shape}")
        print(f"  Type: {type(X)}")
        print(f"  Number of samples: {X.shape[0]}")
        print(f"  Number of features: {X.shape[1]}")
        
        # Show labels
        print(f"\nLabels (1=True, 0=Fake):")
        print(f"  True news: {sum(labels == 1)} articles")
        print(f"  Fake news: {sum(labels == 0)} articles")
        
        # Save the vectorizers
        engineer.save_vectorizers()
        
        # Save features and labels for model training
        print("\nSaving features and labels...")
        from scipy import sparse
        sparse.save_npz('data/X_features.npz', X)
        np.save('data/y_labels.npy', labels)
        print("  Saved features to: data/X_features.npz")
        print("  Saved labels to: data/y_labels.npy")
        
        # Show feature statistics
        print("\nFeature Statistics:")
        if sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        print(f"  Feature matrix min: {X_dense.min():.4f}")
        print(f"  Feature matrix max: {X_dense.max():.4f}")
        print(f"  Feature matrix mean: {X_dense.mean():.4f}")
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING TEST COMPLETE!")
        print("="*60)
        
        return X, labels, engineer
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run preprocessing.py first!")
        return None, None, None

if __name__ == "__main__":
    test_feature_engineering()