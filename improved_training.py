# improved_training.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

class ImprovedFakeNewsDetector:
    def __init__(self):
        print("Improved Fake News Detector Initializing...")
        
    def enhanced_feature_engineering(self, texts):
        """Create enhanced features"""
        print("Creating enhanced features...")
        
        # 1. Basic text features
        features = []
        
        for text in texts:
            text_lower = text.lower()
            words = text_lower.split()
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Sensational words
            sensational_words = ['breaking', 'urgent', 'shocking', 'secret', 'exposed', 
                               'revealed', 'confirmed', 'leaked', 'whistleblower', 'truth',
                               'hidden', 'suppressed', 'wake up', 'they don\'t want you']
            
            # Reliable words
            reliable_words = ['study', 'research', 'according to', 'published', 'journal',
                            'university', 'scientists', 'researchers', 'data', 'analysis',
                            'report', 'confirmed', 'official', 'ministry', 'department']
            
            sensational_count = sum(1 for word in words if any(sw in word for sw in sensational_words))
            reliable_count = sum(1 for word in words if any(rw in word for rw in reliable_words))
            
            # Exaggeration markers
            exaggeration_words = ['completely', 'absolutely', 'totally', 'entirely', 
                                'perfectly', 'extremely', 'incredibly', 'unbelievably']
            exaggeration_count = sum(1 for word in words if word in exaggeration_words)
            
            # ALL CAPS ratio
            all_caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
            
            # Punctuation analysis
            exclamation_ratio = text.count('!') / len(sentences) if sentences else 0
            question_ratio = text.count('?') / len(sentences) if sentences else 0
            
            # Statistical features
            word_count = len(words)
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            avg_sentence_length = word_count / len(sentences) if sentences else 0
            
            # Specific fake news indicators
            conspiracy_terms = ['conspiracy', 'cover-up', 'suppressed', 'hidden truth', 
                              'mainstream media', 'they don\'t want you', 'wake up']
            conspiracy_count = sum(1 for word in words if any(ct in word for ct in conspiracy_terms))
            
            miracle_cure_terms = ['miracle cure', 'instant cure', 'cures instantly', 
                                 'doctors hate', 'simple trick', 'one simple']
            miracle_cure_count = sum(1 for word in words if any(mc in word for mc in miracle_cure_terms))
            
            feature_vector = [
                word_count,
                avg_word_length,
                avg_sentence_length,
                sensational_count,
                reliable_count,
                sensational_count / word_count if word_count > 0 else 0,
                reliable_count / word_count if word_count > 0 else 0,
                exclamation_ratio,
                question_ratio,
                all_caps_words,
                exaggeration_count,
                conspiracy_count,
                miracle_cure_count,
                text.count('!'),  # Total exclamations
                text.count('?'),  # Total questions
                text.count('$'),  # Money mentions
                text.count('%'),  # Percentage mentions
                len([w for w in words if len(w) > 10]),  # Long words
                len([w for w in words if len(w) < 4]),   # Short words
            ]
            
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        # 2. TF-IDF features
        tfidf = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2
        )
        tfidf_features = tfidf.fit_transform(texts)
        
        print(f"Basic features: {features_array.shape}")
        print(f"TF-IDF features: {tfidf_features.shape}")
        
        # Combine features
        from scipy import sparse
        combined_features = sparse.hstack([tfidf_features, sparse.csr_matrix(features_array)])
        
        print(f"Combined features: {combined_features.shape}")
        
        return combined_features, tfidf
    
    def train_improved_model(self, X, y):
        """Train an improved model with better parameters"""
        print("\nTraining improved model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Try different models with better parameters
        models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                C=0.1,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['f1'])
        best_model = results[best_model_name]['model']
        
        print(f"\n✨ Best model: {best_model_name} (F1: {results[best_model_name]['f1']:.4f})")
        
        return best_model, results
    
    def create_stacked_model(self, X, y, base_models):
        """Create a stacked ensemble model"""
        print("\nCreating stacked ensemble...")
        
        from sklearn.model_selection import KFold
        from sklearn.base import BaseEstimator, TransformerMixin, clone
        
        class StackingClassifier(BaseEstimator, TransformerMixin):
            def __init__(self, base_models, meta_model, n_folds=5):
                self.base_models = base_models
                self.meta_model = meta_model
                self.n_folds = n_folds
                
            def fit(self, X, y):
                self.base_models_ = [clone(x) for x in self.base_models]
                self.meta_model_ = clone(self.meta_model)
                
                # Generate meta-features
                kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
                meta_features = np.zeros((X.shape[0], len(self.base_models)))
                
                for i, model in enumerate(self.base_models_):
                    for train_idx, val_idx in kfold.split(X, y):
                        clone_model = clone(model)
                        clone_model.fit(X[train_idx], y[train_idx])
                        meta_features[val_idx, i] = clone_model.predict_proba(X[val_idx])[:, 1]
                
                # Train meta-model
                self.meta_model_.fit(meta_features, y)
                return self
                
            def predict(self, X):
                meta_features = np.column_stack([
                    model.predict_proba(X)[:, 1] for model in self.base_models_
                ])
                return self.meta_model_.predict(meta_features)
                
            def predict_proba(self, X):
                meta_features = np.column_stack([
                    model.predict_proba(X)[:, 1] for model in self.base_models_
                ])
                return self.meta_model_.predict_proba(meta_features)
        
        # Create base models
        base_models = [
            xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
            lgb.LGBMClassifier(n_estimators=100, random_state=42),
            RandomForestClassifier(n_estimators=100, random_state=42)
        ]
        
        # Create meta-model
        meta_model = LogisticRegression(random_state=42)
        
        # Create stacking classifier
        stacking_model = StackingClassifier(base_models=base_models, meta_model=meta_model)
        stacking_model.fit(X, y)
        
        return stacking_model
    
    def save_improved_model(self, model, filename='models/improved_model.pkl'):
        """Save the improved model"""
        import os
        os.makedirs('models', exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model saved to {filename}")

def main():
    """Main function to run improved training"""
    print("="*60)
    print("IMPROVED FAKE NEWS DETECTION TRAINING")
    print("="*60)
    
    # Initialize
    detector = ImprovedFakeNewsDetector()
    
    try:
        # Load data
        print("\nLoading data...")
        df = pd.read_csv('data/news_data.csv')
        print(f"Loaded {len(df)} articles")
        
        # Get texts and labels
        texts = df['text'].fillna('').tolist()
        labels = df['label'].values
        
        # Create enhanced features
        X, tfidf_vectorizer = detector.enhanced_feature_engineering(texts)
        y = labels
        
        # Save the TF-IDF vectorizer
        with open('models/improved_tfidf.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        print("Saved improved TF-IDF vectorizer")
        
        # Train improved model
        best_model, results = detector.train_improved_model(X, y)
        
        # Create and save stacked model
        stacked_model = detector.create_stacked_model(X, y, [
            xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
            lgb.LGBMClassifier(n_estimators=100, random_state=42),
            RandomForestClassifier(n_estimators=100, random_state=42)
        ])
        
        # Save stacked model
        detector.save_improved_model(stacked_model, 'models/stacked_model.pkl')
        
        # Show final comparison
        print("\n" + "="*60)
        print("FINAL MODEL COMPARISON")
        print("="*60)
        
        for name, result in results.items():
            print(f"{name:20} | Accuracy: {result['accuracy']:.4f} | F1: {result['f1']:.4f}")
        
        print("\n" + "="*60)
        print("IMPROVED TRAINING COMPLETE!")
        print("Next: Update app.py to use the improved model")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run data_collection.py first!")

if __name__ == "__main__":
    main()