# Fixed Production Anti-Cheat System
# Save as: production_anticheat_fixed.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import sqlite3
import joblib
import json
import os
from datetime import datetime

class FixedAntiCheatSystem:
    def __init__(self, db_path="anticheat.db"):
        self.db_path = db_path
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.feature_names = []
        
    def load_and_prepare_data(self):
        """Load and prepare data with consistent preprocessing"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            p.steam_id,
            p.username,
            p.vac_banned,
            AVG(CAST(ms.kills AS FLOAT)) as avg_kills,
            AVG(CAST(ms.deaths AS FLOAT)) as avg_deaths,
            AVG(CAST(ms.headshots AS FLOAT)) as avg_headshots,
            AVG(CAST(ms.accuracy AS FLOAT)) as avg_accuracy,
            MAX(CAST(ms.kills AS FLOAT)) as max_kills,
            MIN(CAST(ms.kills AS FLOAT)) as min_kills,
            MAX(CAST(ms.accuracy AS FLOAT)) as max_accuracy,
            MIN(CAST(ms.accuracy AS FLOAT)) as min_accuracy,
            COUNT(ms.id) as total_sessions
        FROM players p
        LEFT JOIN match_stats ms ON p.steam_id = ms.steam_id
        GROUP BY p.steam_id
        HAVING COUNT(ms.id) > 0
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Feature engineering
        features = self.engineer_features(df)
        labels = self.create_labels(df)
        
        return features, labels, df
    
    def engineer_features(self, df):
        """Consistent feature engineering"""
        # Handle missing values first
        df_clean = df.fillna(0)
        
        # Create features
        features = pd.DataFrame()
        features['avg_kills'] = df_clean['avg_kills']
        features['avg_deaths'] = df_clean['avg_deaths']
        features['avg_accuracy'] = df_clean['avg_accuracy']
        features['max_kills'] = df_clean['max_kills']
        features['max_accuracy'] = df_clean['max_accuracy']
        features['total_sessions'] = df_clean['total_sessions']
        
        # Derived features
        features['kd_ratio'] = features['avg_kills'] / np.maximum(features['avg_deaths'], 0.1)
        features['headshot_ratio'] = df_clean['avg_headshots'] / np.maximum(features['avg_kills'], 0.1)
        features['kill_variance'] = (features['max_kills'] - df_clean['min_kills']) / np.maximum(features['avg_kills'], 0.1)
        features['accuracy_variance'] = (features['max_accuracy'] - df_clean['min_accuracy']) / np.maximum(features['avg_accuracy'], 0.1)
        
        # Anomaly indicators
        features['high_accuracy'] = (features['avg_accuracy'] > 60).astype(int)
        features['extreme_headshots'] = (features['headshot_ratio'] > 0.7).astype(int)
        features['high_kd'] = (features['kd_ratio'] > 4).astype(int)
        features['perfect_accuracy'] = (features['max_accuracy'] > 90).astype(int)
        
        # Combined score
        features['suspicion_score'] = (
            features['high_accuracy'] * 0.3 +
            features['extreme_headshots'] * 0.3 +
            features['high_kd'] * 0.2 +
            features['perfect_accuracy'] * 0.2
        )
        
        # Fill any remaining NaN values
        features = features.fillna(0)
        
        self.feature_names = features.columns.tolist()
        return features
    
    def create_labels(self, df):
        """Create ground truth labels"""
        labels = []
        for _, row in df.iterrows():
            if row['vac_banned']:
                labels.append(1)
            else:
                # Statistical impossibility check
                acc = row.get('avg_accuracy', 0) or 0
                kd = (row.get('avg_kills', 0) or 0) / max(row.get('avg_deaths', 1) or 1, 1)
                hs = (row.get('avg_headshots', 0) or 0) / max(row.get('avg_kills', 1) or 1, 1)
                
                if acc > 70 or (kd > 4 and hs > 0.7) or acc > 80:
                    labels.append(1)
                else:
                    labels.append(0)
        return np.array(labels)
    
    def train_models(self, X, y):
        """Train ensemble models with fixed preprocessing"""
        print(f"Training on {len(X)} samples with {X.shape[1]} features")
        print(f"Positive class: {y.sum()}/{len(y)} ({y.sum()/len(y)*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features - fit on train, transform both
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection - fit on train, transform both
        self.feature_selector = SelectKBest(f_classif, k=min(10, X_train.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        print(f"Selected {X_train_selected.shape[1]} features")
        
        # Train models
        models_config = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
        }
        
        results = {}
        for name, model in models_config.items():
            print(f"Training {name}...")
            model.fit(X_train_selected, y_train)
            
            train_acc = model.score(X_train_selected, y_train)
            test_acc = model.score(X_test_selected, y_test)
            
            print(f"  {name}: Train {train_acc:.3f}, Test {test_acc:.3f}")
            
            self.models[name] = model
            results[name] = test_acc
        
        # Comprehensive evaluation
        self.evaluate_models(X_test_selected, y_test)
        
        return results
    
    def predict(self, X):
        """Make predictions with proper preprocessing pipeline"""
        # Apply same preprocessing as training
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_selected)
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_selected)[:, 1]
            else:
                prob = pred.astype(float)
            
            predictions[name] = pred
            probabilities[name] = prob
        
        # Ensemble prediction (average probabilities)
        ensemble_prob = np.mean(list(probabilities.values()), axis=0)
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        return ensemble_pred, ensemble_prob, predictions
    
    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation"""
        ensemble_pred, ensemble_prob, individual_preds = self.predict_preprocessed(X_test)
        
        # Calculate metrics
        accuracy = np.mean(ensemble_pred == y_test)
        
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, ensemble_prob)
            cm = confusion_matrix(y_test, ensemble_pred)
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                print(f"\nEVALUATION RESULTS:")
                print(f"Accuracy: {accuracy:.3f}")
                print(f"Precision: {precision:.3f}")
                print(f"Recall: {recall:.3f}")
                print(f"False Positive Rate: {fpr:.3f}")
                print(f"AUC: {auc:.3f}")
                print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    def predict_preprocessed(self, X_preprocessed):
        """Predict on already preprocessed data"""
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_preprocessed)
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_preprocessed)[:, 1]
            else:
                prob = pred.astype(float)
            
            predictions[name] = pred
            probabilities[name] = prob
        
        ensemble_prob = np.mean(list(probabilities.values()), axis=0)
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        return ensemble_pred, ensemble_prob, predictions
    
    def save_models(self, save_dir="fixed_models"):
        """Save trained models and preprocessors"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{save_dir}/{name}_model.joblib")
        
        # Save preprocessors
        joblib.dump(self.scaler, f"{save_dir}/scaler.joblib")
        joblib.dump(self.feature_selector, f"{save_dir}/feature_selector.joblib")
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'model_names': list(self.models.keys()),
            'trained_at': datetime.now().isoformat()
        }
        
        with open(f"{save_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Models saved to {save_dir}/")

def main():
    print("FIXED PRODUCTION ANTI-CHEAT SYSTEM")
    print("=" * 50)
    
    system = FixedAntiCheatSystem()
    
    # Load and prepare data
    X, y, df = system.load_and_prepare_data()
    print(f"Dataset: {len(X)} players, {y.sum()} cheaters")
    
    # Train models
    results = system.train_models(X, y)
    
    # Save models
    system.save_models()
    
    print(f"\nTRAINING COMPLETE")
    print("Models saved and ready for deployment")
    
    return system

if __name__ == "__main__":
    main()