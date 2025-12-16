# model_training.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import pickle
import os
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

class FakeNewsModelTrainer:
    def __init__(self, random_state=42):
        """
        Initialize model trainer
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
        # Set style for plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        print("Fake News Model Trainer Initialized!")
    
    def load_data(self):
        """
        Load features and labels
        """
        print("Loading features and labels...")
        
        try:
            # Load sparse features
            X = sparse.load_npz('data/X_features.npz')
            y = np.load('data/y_labels.npy')
            
            print(f"Features shape: {X.shape}")
            print(f"Labels shape: {y.shape}")
            print(f"True news samples: {sum(y == 1)}")
            print(f"Fake news samples: {sum(y == 0)}")
            
            return X, y
            
        except FileNotFoundError:
            print("Error: Feature files not found!")
            print("Please run feature_engineering.py first.")
            return None, None
    
    def train_models(self, X, y):
        """
        Train multiple machine learning models
        """
        print("\n" + "="*60)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.25, 
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        # Define models to train
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='linear',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'Naive Bayes': MultinomialNB(),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=1
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Store results
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'test_indices': (X_test, y_test)
                }
                
                print(f"  Accuracy:  {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall:    {recall:.4f}")
                print(f"  F1-Score:  {f1:.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)[:100]}...")
                results[name] = None
        
        self.models = results
        
        # Find best model based on F1 score
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            self.best_model_name = max(valid_results, key=lambda x: valid_results[x]['f1_score'])
            self.best_model = valid_results[self.best_model_name]['model']
            print(f"\n✨ Best Model: {self.best_model_name} (F1-Score: {valid_results[self.best_model_name]['f1_score']:.4f})")
        
        return results
    
    def create_ensemble(self, X, y):
        """
        Create an ensemble model using voting
        """
        print("\n" + "-"*40)
        print("CREATING ENSEMBLE MODEL")
        print("-"*40)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )
        
        # Create base models
        estimators = [
            ('xgboost', xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )),
            ('lr', LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            ))
        ]
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        
        # Train ensemble
        print(f"Training ensemble with {len(estimators)} models...")
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"Ensemble Accuracy: {accuracy:.4f}")
        print(f"Ensemble F1-Score: {f1:.4f}")
        
        return ensemble
    
    def evaluate_and_save(self):
        """
        Evaluate models and save the best ones
        """
        print("\n" + "="*60)
        print("EVALUATING AND SAVING MODELS")
        print("="*60)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('static', exist_ok=True)
        
        # Save each model
        for name, result in self.models.items():
            if result is not None:
                filename = f"models/{name.lower().replace(' ', '_')}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(result['model'], f)
                print(f"Saved {name} to {filename}")
        
        # Save best model
        if self.best_model is not None:
            with open('models/best_model.pkl', 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"\nSaved best model ({self.best_model_name}) to models/best_model.pkl")
        
        print("\nAll models saved successfully!")

def main():
    """
    Main function to run the model training pipeline
    """
    print("="*60)
    print("FAKE NEWS DETECTION - MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Initialize trainer
    trainer = FakeNewsModelTrainer(random_state=42)
    
    # Step 1: Load data
    X, y = trainer.load_data()
    if X is None:
        return
    
    # Step 2: Train individual models
    results = trainer.train_models(X, y)
    
    # Step 3: Create and evaluate ensemble
    ensemble = trainer.create_ensemble(X, y)
    
    # Save ensemble model
    with open('models/ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    print(f"\nSaved ensemble model to models/ensemble_model.pkl")
    
    # Step 4: Evaluate and save all models
    trainer.evaluate_and_save()
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the 'models' folder for saved models")
    print("2. Run app.py to start the web application")
    print("3. Open browser and go to: http://localhost:5000")

if __name__ == "__main__":
    main()