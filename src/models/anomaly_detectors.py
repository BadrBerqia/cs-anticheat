# Anti-Cheat ML Models
# Save as: src\models\anomaly_detectors.py

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sqlite3
import os
import sys
import joblib
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.database import AntiCheatDatabase

class AntiCheatMLModels:
    def __init__(self, db_path="anticheat.db"):
        self.db = AntiCheatDatabase(db_path)
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
    def load_player_data(self):
        """Load player data from database for training"""
        conn = self.db.get_connection()
        
        # Get all players with their basic stats
        query = """
        SELECT 
            p.steam_id,
            p.username,
            p.total_playtime,
            p.vac_banned,
            COUNT(ms.id) as total_matches,
            AVG(ms.kills) as avg_kills,
            AVG(ms.deaths) as avg_deaths,
            AVG(ms.headshots) as avg_headshots,
            AVG(ms.accuracy) as avg_accuracy,
            AVG(ms.anomaly_score) as avg_anomaly_score,
            MAX(ms.kills) as max_kills,
            MAX(ms.headshots) as max_headshots,
            MAX(ms.accuracy) as max_accuracy,
            MIN(ms.deaths) as min_deaths
        FROM players p
        LEFT JOIN match_stats ms ON p.steam_id = ms.steam_id
        GROUP BY p.steam_id
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"📊 Loaded {len(df)} players from database")
        return df
    
    def engineer_features(self, df):
        """Create features for machine learning"""
        features = df.copy()
        
        # Handle missing values
        features = features.fillna(0)
        
        # Create derived features
        features['kd_ratio'] = features['avg_kills'] / (features['avg_deaths'] + 0.1)  # Avoid division by zero
        features['headshot_ratio'] = features['avg_headshots'] / (features['avg_kills'] + 0.1)
        features['max_kill_efficiency'] = features['max_kills'] / (features['total_matches'] + 1)
        features['consistency_score'] = features['avg_accuracy'] / (features['max_accuracy'] + 0.1)
        features['death_avoidance'] = 1 / (features['min_deaths'] + 1)
        
        # Playtime features
        features['experience_factor'] = np.log1p(features['total_playtime'])
        features['matches_per_hour'] = features['total_matches'] / (features['total_playtime'] / 60 + 1)
        
        # Statistical impossibility indicators
        features['impossible_headshot_rate'] = (features['headshot_ratio'] > 0.8).astype(int)
        features['impossible_accuracy'] = (features['avg_accuracy'] > 60).astype(int)
        features['impossible_kd'] = (features['kd_ratio'] > 5).astype(int)
        
        # Behavioral anomalies
        features['perfect_games_indicator'] = (features['max_accuracy'] > 95).astype(int)
        features['no_death_games'] = (features['min_deaths'] == 0).astype(int)
        
        # Select feature columns for ML
        feature_cols = [
            'total_playtime', 'avg_kills', 'avg_deaths', 'avg_headshots', 'avg_accuracy',
            'max_kills', 'max_headshots', 'max_accuracy', 'kd_ratio', 'headshot_ratio',
            'max_kill_efficiency', 'consistency_score', 'death_avoidance', 'experience_factor',
            'matches_per_hour', 'impossible_headshot_rate', 'impossible_accuracy', 
            'impossible_kd', 'perfect_games_indicator', 'no_death_games'
        ]
        
        self.feature_names = feature_cols
        return features[feature_cols].fillna(0)
    
    def create_labels(self, df):
        """Create ground truth labels for supervised learning"""
        # Create labels based on VAC bans and high anomaly scores
        labels = []
        
        for _, row in df.iterrows():
            if row['vac_banned']:
                labels.append(1)  # Definitely cheating
            elif row['avg_anomaly_score'] > 0.7:
                labels.append(1)  # Likely cheating based on statistical analysis
            else:
                labels.append(0)  # Likely legitimate
        
        return np.array(labels)
    
    def train_isolation_forest(self, X, contamination=0.1):
        """Train Isolation Forest for anomaly detection"""
        print("🌳 Training Isolation Forest...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        model.fit(X_scaled)
        
        # Store model and scaler
        self.models['isolation_forest'] = model
        self.scalers['isolation_forest'] = scaler
        
        # Get anomaly scores
        anomaly_scores = model.decision_function(X_scaled)
        predictions = model.predict(X_scaled)
        
        print(f"✅ Isolation Forest trained")
        print(f"   Detected {np.sum(predictions == -1)} anomalies out of {len(predictions)} players")
        
        return predictions, anomaly_scores
    
    def train_one_class_svm(self, X, nu=0.1):
        """Train One-Class SVM for anomaly detection"""
        print("🔍 Training One-Class SVM...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        model.fit(X_scaled)
        
        # Store model and scaler
        self.models['one_class_svm'] = model
        self.scalers['one_class_svm'] = scaler
        
        # Get predictions
        predictions = model.predict(X_scaled)
        anomaly_scores = model.decision_function(X_scaled)
        
        print(f"✅ One-Class SVM trained")
        print(f"   Detected {np.sum(predictions == -1)} anomalies out of {len(predictions)} players")
        
        return predictions, anomaly_scores
    
    def train_statistical_detector(self, X, df):
        """Train statistical rule-based detector"""
        print("📊 Training Statistical Detector...")
        
        def statistical_predict(features, player_data):
            scores = []
            
            for i in range(len(features)):
                score = 0.0
                row = player_data.iloc[i]
                
                # Handle None/NaN values safely
                avg_accuracy = row.get('avg_accuracy') or 0
                headshot_ratio = row.get('headshot_ratio') or 0
                kd_ratio = row.get('kd_ratio') or 0
                max_accuracy = row.get('max_accuracy') or 0
                death_avoidance = row.get('death_avoidance') or 0
                
                # VAC banned players
                if row['vac_banned']:
                    score = 1.0
                else:
                    # Statistical impossibilities (only check if we have data)
                    if avg_accuracy > 65:  # >65% accuracy is suspicious
                        score += 0.4
                    if headshot_ratio > 0.7:  # >70% headshots
                        score += 0.3
                    if kd_ratio > 6:  # Very high K/D
                        score += 0.2
                    if max_accuracy > 90:  # Perfect accuracy games
                        score += 0.3
                    if death_avoidance > 0.5:  # Almost never dies
                        score += 0.2
                
                scores.append(min(score, 1.0))
            
            return np.array(scores)
        
        # Get statistical scores
        statistical_scores = statistical_predict(X, df)
        predictions = (statistical_scores > 0.5).astype(int)
        predictions[predictions == 0] = 1  # Convert to -1/1 format for consistency
        predictions[predictions == 1] = -1
        predictions[predictions == 0] = 1
        
        # Store as a simple function
        self.models['statistical'] = statistical_predict
        
        print(f"✅ Statistical Detector trained")
        print(f"   Detected {np.sum(predictions == -1)} anomalies out of {len(predictions)} players")
        
        return predictions, statistical_scores
    
    def ensemble_predict(self, X, df=None):
        """Combine predictions from multiple models"""
        if not self.models:
            raise ValueError("No models trained yet!")
        
        predictions = {}
        scores = {}
        
        # Isolation Forest
        if 'isolation_forest' in self.models:
            X_scaled = self.scalers['isolation_forest'].transform(X)
            predictions['isolation_forest'] = self.models['isolation_forest'].predict(X_scaled)
            scores['isolation_forest'] = self.models['isolation_forest'].decision_function(X_scaled)
        
        # One-Class SVM
        if 'one_class_svm' in self.models:
            X_scaled = self.scalers['one_class_svm'].transform(X)
            predictions['one_class_svm'] = self.models['one_class_svm'].predict(X_scaled)
            scores['one_class_svm'] = self.models['one_class_svm'].decision_function(X_scaled)
        
        # Statistical detector
        if 'statistical' in self.models and df is not None:
            scores['statistical'] = self.models['statistical'](X, df)
            predictions['statistical'] = (scores['statistical'] > 0.5).astype(int)
            predictions['statistical'][predictions['statistical'] == 0] = 1
            predictions['statistical'][predictions['statistical'] == 1] = -1
            predictions['statistical'][predictions['statistical'] == 0] = 1
        
        # Ensemble voting (majority vote)
        if len(predictions) > 1:
            pred_array = np.array(list(predictions.values()))
            ensemble_pred = np.apply_along_axis(lambda x: 1 if np.sum(x == 1) > np.sum(x == -1) else -1, axis=0, arr=pred_array)
        else:
            ensemble_pred = list(predictions.values())[0]
        
        # Average scores for ensemble confidence
        if scores:
            # Normalize scores to 0-1 range
            normalized_scores = {}
            for name, score_array in scores.items():
                min_score, max_score = np.min(score_array), np.max(score_array)
                if max_score > min_score:
                    normalized_scores[name] = (score_array - min_score) / (max_score - min_score)
                else:
                    normalized_scores[name] = np.zeros_like(score_array)
            
            ensemble_scores = np.mean(list(normalized_scores.values()), axis=0)
        else:
            ensemble_scores = np.zeros(len(X))
        
        return ensemble_pred, ensemble_scores, predictions, scores
    
    def save_models(self, model_dir="data/models"):
        """Save trained models to disk"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models and scalers
        for name, model in self.models.items():
            if name != 'statistical':  # Skip the function-based model
                joblib.dump(model, os.path.join(model_dir, f"{name}_model.joblib"))
        
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(model_dir, f"{name}_scaler.joblib"))
        
        # Save feature names
        with open(os.path.join(model_dir, "feature_names.json"), 'w') as f:
            json.dump(self.feature_names, f)
        
        print(f"✅ Models saved to {model_dir}")
    
    def load_models(self, model_dir="data/models"):
        """Load trained models from disk"""
        # Load feature names
        with open(os.path.join(model_dir, "feature_names.json"), 'r') as f:
            self.feature_names = json.load(f)
        
        # Load models and scalers
        model_files = ['isolation_forest_model.joblib', 'one_class_svm_model.joblib']
        scaler_files = ['isolation_forest_scaler.joblib', 'one_class_svm_scaler.joblib']
        
        for model_file in model_files:
            if os.path.exists(os.path.join(model_dir, model_file)):
                name = model_file.replace('_model.joblib', '')
                self.models[name] = joblib.load(os.path.join(model_dir, model_file))
        
        for scaler_file in scaler_files:
            if os.path.exists(os.path.join(model_dir, scaler_file)):
                name = scaler_file.replace('_scaler.joblib', '')
                self.scalers[name] = joblib.load(os.path.join(model_dir, scaler_file))
        
        print(f"✅ Models loaded from {model_dir}")


def main():
    """Main training function"""
    print("🤖 Starting Anti-Cheat ML Training...")
    
    # Initialize ML system
    ml_system = AntiCheatMLModels()
    
    # Load data
    df = ml_system.load_player_data()
    
    if len(df) == 0:
        print("❌ No player data found! Please collect some data first.")
        return
    
    print(f"📊 Dataset size: {len(df)} players")
    print(f"🚨 VAC banned players: {df['vac_banned'].sum()}")
    
    # Engineer features
    X = ml_system.engineer_features(df)
    print(f"🔧 Created {X.shape[1]} features")
    print(f"📋 Features: {ml_system.feature_names}")
    
    # Train models
    print("\n🚀 Training ML Models...")
    
    # Train Isolation Forest
    if_pred, if_scores = ml_system.train_isolation_forest(X, contamination=0.15)
    
    # Train One-Class SVM
    svm_pred, svm_scores = ml_system.train_one_class_svm(X, nu=0.15)
    
    # Train Statistical Detector
    stat_pred, stat_scores = ml_system.train_statistical_detector(X, df)
    
    # Test ensemble
    print("\n🎯 Testing Ensemble Model...")
    ensemble_pred, ensemble_scores, individual_preds, individual_scores = ml_system.ensemble_predict(X, df)
    
    # Analyze results
    print(f"\n📈 RESULTS SUMMARY:")
    print(f"Isolation Forest anomalies: {np.sum(if_pred == -1)}")
    print(f"One-Class SVM anomalies: {np.sum(svm_pred == -1)}")
    print(f"Statistical anomalies: {np.sum(stat_pred == -1)}")
    print(f"Ensemble anomalies: {np.sum(ensemble_pred == -1)}")
    
    # Show most suspicious players
    suspicious_indices = np.where(ensemble_pred == -1)[0]
    if len(suspicious_indices) > 0:
        print(f"\n🚨 Most suspicious players:")
        for i in suspicious_indices[:5]:
            player = df.iloc[i]
            score = ensemble_scores[i]
            print(f"  - {player['username']}: {score:.3f} (VAC: {player['vac_banned']})")
    
    # Save models
    ml_system.save_models()
    print("\n✅ Training complete! Models saved and ready for deployment.")


if __name__ == "__main__":
    main()