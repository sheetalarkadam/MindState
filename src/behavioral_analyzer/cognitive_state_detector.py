"""
Cognitive State Detection System
Uses machine learning to classify cognitive states from behavioral patterns
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple
import joblib
import os
from datetime import datetime

class BehavioralFeatureExtractor:
    """Extract meaningful features from raw behavioral events"""
    
    def __init__(self, window_size_seconds: int = 60):
        self.window_size = window_size_seconds
        
    def extract_session_features(self, session_events: pd.DataFrame) -> Dict:
        """Extract aggregated features from a session or time window"""
        
        if len(session_events) == 0:
            return {}
        
        features = {}
        
        # Basic statistical features for key metrics
        numeric_cols = ['typing_speed', 'typing_errors', 'mouse_precision', 
                       'click_frequency', 'scroll_smoothness', 'pause_duration', 
                       'session_focus', 'app_switches']
        
        for col in numeric_cols:
            if col in session_events.columns:
                # Central tendency
                features[f'{col}_mean'] = session_events[col].mean()
                features[f'{col}_median'] = session_events[col].median()
                
                # Variability
                features[f'{col}_std'] = session_events[col].std()
                features[f'{col}_variance'] = session_events[col].var()
                
                # Distribution shape
                features[f'{col}_skew'] = session_events[col].skew()
                features[f'{col}_kurtosis'] = session_events[col].kurtosis()
                
                # Range
                features[f'{col}_min'] = session_events[col].min()
                features[f'{col}_max'] = session_events[col].max()
                features[f'{col}_range'] = session_events[col].max() - session_events[col].min()
        
        # Temporal patterns (if we have timestamps)
        if 'session_minute' in session_events.columns and len(session_events) > 1:
            # Trend analysis - are metrics getting better or worse over time?
            time_points = session_events['session_minute'].values
            
            for col in ['typing_speed', 'session_focus', 'mouse_precision']:
                if col in session_events.columns:
                    values = session_events[col].values
                    # Simple linear trend
                    trend_slope = np.polyfit(time_points, values, 1)[0]
                    features[f'{col}_trend'] = trend_slope
        
        # Derived cognitive load indicators
        if all(col in session_events.columns for col in ['typing_speed', 'typing_errors']):
            # Speed vs accuracy trade-off
            features['speed_accuracy_ratio'] = (
                session_events['typing_speed'].mean() / 
                (session_events['typing_errors'].mean() + 0.01)  # Avoid division by zero
            )
        
        if all(col in session_events.columns for col in ['session_focus', 'app_switches']):
            # Attention vs distraction balance
            features['focus_distraction_ratio'] = (
                session_events['session_focus'].mean() / 
                (session_events['app_switches'].mean() + 0.01)
            )
        
        if all(col in session_events.columns for col in ['mouse_precision', 'click_frequency']):
            # Motor control efficiency
            features['motor_efficiency'] = (
                session_events['mouse_precision'].mean() / 
                (session_events['click_frequency'].mean() + 0.1)
            )
        
        # Session-level features
        features['session_length'] = len(session_events)
        features['total_duration'] = session_events['session_minute'].max() if 'session_minute' in session_events.columns else 0
        
        # Replace any NaN values with 0
        for key, value in features.items():
            if pd.isna(value):
                features[key] = 0.0
        
        return features

class CognitiveStateClassifier:
    """Machine learning classifier for cognitive states"""
    
    def __init__(self):
        # Ensemble of classifiers
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=15, 
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=42
            )
        }
        
        self.feature_extractor = BehavioralFeatureExtractor()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        # State mappings
        self.state_encoder = {
            'flow_state': 0, 'stress_state': 1, 'recovery_mode': 2, 
            'discovery_mode': 3, 'learning_mode': 4
        }
        self.state_decoder = {v: k for k, v in self.state_encoder.items()}
    
    def prepare_training_data(self, behavioral_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Convert raw behavioral data into training features and labels"""
        
        print("Preparing training data...")
        
        features_list = []
        labels = []
        
        # Group by session and extract features for each session
        session_groups = behavioral_data.groupby(['session_id', 'cognitive_state'])
        
        print(f"Processing {len(session_groups)} sessions...")
        
        for (session_id, state), session_data in session_groups:
            # Extract features for this session
            session_features = self.feature_extractor.extract_session_features(session_data)
            
            if session_features:  # Only add if we got valid features
                features_list.append(session_features)
                labels.append(state)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        labels_series = pd.Series(labels)
        
        # Store feature names for later use
        self.feature_names = features_df.columns.tolist()
        
        print(f"Extracted {len(features_df)} feature vectors with {len(self.feature_names)} features each")
        print(f"Label distribution:\n{labels_series.value_counts()}")
        
        return features_df, labels_series
    
    def train(self, behavioral_data: pd.DataFrame) -> Dict:
        """Train the cognitive state classification models"""
        
        print("🧠 Training Cognitive State Classifier...")
        print("=" * 50)
        
        # Prepare training data
        X, y = self.prepare_training_data(behavioral_data)
        
        if len(X) == 0:
            raise ValueError("No valid training data generated")
        
        # Encode labels
        y_encoded = y.map(self.state_encoder)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train each model
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Training accuracy: {train_score:.3f}")
            print(f"  Test accuracy: {test_score:.3f}")
            print(f"  Cross-val: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Detailed evaluation on best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        best_model = self.models[best_model_name]
        
        print(f"\n📊 Detailed evaluation of best model ({best_model_name}):")
        y_pred = best_model.predict(X_test_scaled)
        
        # Classification report
        state_names = [self.state_decoder[i] for i in sorted(self.state_decoder.keys())]
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=state_names))
        
        # Feature importance (for Random Forest)
        if hasattr(best_model, 'feature_importances_'):
            print(f"\n�� Top 10 Most Important Features:")
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(feature_importance.head(10).to_string(index=False))
        
        self.is_trained = True
        
        print(f"\n✅ Training complete! Best model: {best_model_name}")
        
        return results
    
    def predict(self, behavioral_data: pd.DataFrame) -> Dict:
        """Predict cognitive state from behavioral data"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract features
        features = self.feature_extractor.extract_session_features(behavioral_data)
        if not features:
            return {'state': 'unknown', 'confidence': 0.0}
        
        # Convert to DataFrame with correct feature order
        feature_df = pd.DataFrame([features])[self.feature_names]
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_df)
        
        # Get predictions from all models
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            pred = model.predict(feature_scaled)[0]
            pred_proba = model.predict_proba(feature_scaled)[0]
            
            predictions[name] = self.state_decoder[pred]
            confidences[name] = max(pred_proba)
        
        # Use the best performing model for final prediction
        # (In practice, you might want to ensemble these)
        best_model_name = 'random_forest'  # or determine dynamically
        
        return {
            'state': predictions[best_model_name],
            'confidence': confidences[best_model_name],
            'all_predictions': predictions,
            'all_confidences': confidences
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'state_encoder': self.state_encoder,
            'state_decoder': self.state_decoder
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.state_encoder = model_data['state_encoder']
        self.state_decoder = model_data['state_decoder']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

def demo_cognitive_classification():
    """Demonstrate the cognitive state classification"""
    
    print("🧠 MindState Cognitive State Classification Demo")
    print("=" * 55)
    
    # Import the behavior generator
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from behavior_simulator import BehaviorDataGenerator
    
    # Generate training data
    print("\n1. Generating behavioral training data...")
    generator = BehaviorDataGenerator()
    dataset = generator.generate_dataset(sessions_per_state=30)  # Moderate size for demo
    
    # Train classifier
    print("\n2. Training cognitive state classifier...")
    classifier = CognitiveStateClassifier()
    results = classifier.train(dataset)
    
    print(f"\n3. Model Performance Summary:")
    for model_name, metrics in results.items():
        print(f"  {model_name}: {metrics['test_accuracy']:.3f} test accuracy")
    
    # Test on new data
    print(f"\n4. Testing on new behavioral sessions...")
    
    # Generate test sessions for each state
    test_states = ['flow_state', 'stress_state', 'recovery_mode']
    
    for state in test_states:
        print(f"\n  Testing {state}:")
        test_session = generator.generate_session_data(state, duration_minutes=5)
        test_df = pd.DataFrame(test_session['events'])
        
        prediction = classifier.predict(test_df)
        
        print(f"    Predicted: {prediction['state']} (confidence: {prediction['confidence']:.2f})")
        print(f"    Actual: {state}")
        print(f"    ✅ Correct!" if prediction['state'] == state else "❌ Incorrect")
    
    print(f"\n🎉 Cognitive state classification demo complete!")
    print(f"🚀 Ready to build real-time monitoring system!")

if __name__ == "__main__":
    demo_cognitive_classification()
