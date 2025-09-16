# Production-Level Anti-Cheat ML System
# Save as: production_anticheat.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import sqlite3
import joblib
from datetime import datetime
import json
import os

class ProductionAntiCheatSystem:
    def __init__(self, db_path="anticheat.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.feature_names = []
        
    def load_production_dataset(self):
        """Load large-scale dataset"""
        conn = sqlite3.connect(self.db_path)
        
        # Load all player data
        query = """
        SELECT 
            p.steam_id,
            p.username,
            p.total_playtime,
            p.vac_banned,
            -- Basic stats
            AVG(CAST(ms.kills AS FLOAT)) as avg_kills,
            AVG(CAST(ms.deaths AS FLOAT)) as avg_deaths,
            AVG(CAST(ms.headshots AS FLOAT)) as avg_headshots,
            AVG(CAST(ms.accuracy AS FLOAT)) as avg_accuracy,
            
            -- Advanced aggregations
            MAX(CAST(ms.kills AS FLOAT)) as max_kills,
            MIN(CAST(ms.kills AS FLOAT)) as min_kills,
            MAX(CAST(ms.accuracy AS FLOAT)) as max_accuracy,
            MIN(CAST(ms.accuracy AS FLOAT)) as min_accuracy,
            
            -- Count features
            COUNT(ms.id) as total_sessions
            
        FROM players p
        LEFT JOIN match_stats ms ON p.steam_id = ms.steam_id
        GROUP BY p.steam_id
        HAVING COUNT(ms.id) > 0
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"Loaded dataset: {len(df)} players")
        return df
    
    def engineer_advanced_features(self, df):
        """Create sophisticated features for ML"""
        features = df.copy()
        
        # Handle missing values
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            features[col] = features[col].fillna(features[col].median())
        
        # Basic derived features
        features['kd_ratio'] = features['avg_kills'] / np.maximum(features['avg_deaths'], 0.1)
        features['headshot_ratio'] = features['avg_headshots'] / np.maximum(features['avg_kills'], 0.1)
        
        # Performance variance features
        features['kill_variance'] = (features['max_kills'] - features['min_kills']) / np.maximum(features['avg_kills'], 0.1)
        features['accuracy_variance'] = (features['max_accuracy'] - features['min_accuracy']) / np.maximum(features['avg_accuracy'], 0.1)
        
        # Extreme performance indicators
        features['extreme_accuracy'] = (features['max_accuracy'] > 90).astype(int)
        features['extreme_kills'] = (features['max_kills'] > features['avg_kills'] * 2.5).astype(int)
        features['perfect_accuracy'] = (features['max_accuracy'] > 95).astype(int)
        
        # Activity features
        features['playtime_per_session'] = features['total_playtime'] / np.maximum(features['total_sessions'], 1)
        features['sessions_intensity'] = features['total_sessions'] / np.maximum(features['total_playtime'] / 60, 1)
        
        # Statistical impossibility scores
        features['accuracy_impossibility'] = np.where(features['avg_accuracy'] > 70, 1, 0)
        features['headshot_impossibility'] = np.where(features['headshot_ratio'] > 0.8, 1, 0)
        features['kd_impossibility'] = np.where(features['kd_ratio'] > 5, 1, 0)
        
        # Combined suspicion score
        features['combined_suspicion'] = (
            features['accuracy_impossibility'] * 0.4 +
            features['headshot_impossibility'] * 0.3 +
            features['kd_impossibility'] * 0.3
        )
        
        # Behavioral consistency
        features['high_consistency'] = np.where(
            (features['accuracy_variance'] < 0.2) & (features['avg_accuracy'] > 60), 1, 0
        )
        
        # Select feature columns
        feature_cols = [
            # Basic performance
            'avg_kills', 'avg_deaths', 'avg_accuracy', 'kd_ratio', 'headshot_ratio',
            
            # Variance metrics  
            'kill_variance', 'accuracy_variance',
            
            # Extreme indicators
            'extreme_accuracy', 'extreme_kills', 'perfect_accuracy',
            
            # Impossibility indicators
            'accuracy_impossibility', 'headshot_impossibility', 'kd_impossibility',
            'combined_suspicion', 'high_consistency',
            
            # Activity patterns
            'total_sessions', 'playtime_per_session', 'sessions_intensity',
            
            # Statistical features
            'max_kills', 'max_accuracy', 'min_kills', 'min_accuracy'
        ]
        
        self.feature_names = feature_cols
        return features[feature_cols].fillna(0)
    
    def create_ground_truth_labels(self, df):
        """Create ground truth labels"""
        labels = []
        
        for _, row in df.iterrows():
            # Primary: VAC bans
            if row['vac_banned']:
                labels.append(1)
                continue
            
            # Secondary: Statistical impossibilities
            impossibility_score = 0
            
            if row['avg_accuracy'] > 70:
                impossibility_score += 2
            elif row['avg_accuracy'] > 60:
                impossibility_score += 1
                
            if row['max_accuracy'] > 90:
                impossibility_score += 1
                
            kd_ratio = row['avg_kills'] / max(row['avg_deaths'], 1)
            headshot_ratio = row['avg_headshots'] / max(row['avg_kills'], 1)
            
            if kd_ratio > 5:
                impossibility_score += 2
            elif kd_ratio > 3:
                impossibility_score += 1
                
            if headshot_ratio > 0.8:
                impossibility_score += 2
            elif headshot_ratio > 0.6:
                impossibility_score += 1
                
            # Multiple impossibilities = likely cheater
            if impossibility_score >= 4:
                labels.append(1)
            else:
                labels.append(0)
        
        return np.array(labels)
    
    def train_ensemble_models(self, X, y):
        """Train multiple advanced ML models"""
        print("Training ensemble of ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Positive class (cheaters): {y_train.sum()} / {y_test.sum()}")
        
        # Feature scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        # Feature selection
        selector = SelectKBest(f_classif, k=min(12, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        self.feature_selector = selector
        
        print(f"Selected {X_train_selected.shape[1]} features out of {X_train.shape[1]}")
        
        # Train models
        models_to_train = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42
            ),
        }
        
        # Train supervised models
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train_selected, y_train)
                self.models[name] = model
                
                # Quick evaluation
                train_acc = model.score(X_train_selected, y_train)
                test_acc = model.score(X_test_selected, y_test)
                print(f"  Train accuracy: {train_acc:.3f}")
                print(f"  Test accuracy: {test_acc:.3f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
        
        # Train unsupervised model
        print("Training Isolation Forest...")
        try:
            iso_model = IsolationForest(contamination=0.15, random_state=42)
            iso_model.fit(X_train_selected)
            self.models['isolation_forest'] = iso_model
            
            # Evaluate isolation forest
            iso_pred = (iso_model.predict(X_test_selected) == -1).astype(int)
            iso_acc = np.mean(iso_pred == y_test)
            print(f"  Isolation Forest accuracy: {iso_acc:.3f}")
            
        except Exception as e:
            print(f"  Error training Isolation Forest: {e}")
        
        return X_test_selected, y_test
    
    def ensemble_predict(self, X):
        """Make ensemble predictions"""
        if not self.models:
            raise ValueError("No models trained!")
        
        # Preprocess
        X_scaled = self.scalers['main'].transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                if name == 'isolation_forest':
                    pred = (model.predict(X_selected) == -1).astype(int)
                    prob = model.decision_function(X_selected)
                    # Normalize scores to 0-1
                    prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
                else:
                    pred = model.predict(X_selected)
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_selected)[:, 1]
                    else:
                        prob = pred.astype(float)
                
                predictions[name] = pred
                probabilities[name] = prob
                
            except Exception as e:
                print(f"Error in {name} prediction: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions!")
        
        # Ensemble voting (simple average)
        ensemble_prob = np.mean(list(probabilities.values()), axis=0)
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        return ensemble_pred, ensemble_prob, predictions, probabilities
    
    def comprehensive_evaluation(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\nEvaluating ensemble model...")
        
        try:
            pred, prob, individual_preds, individual_probs = self.ensemble_predict(X_test)
            
            # Overall metrics
            accuracy = np.mean(pred == y_test)
            
            print(f"\n=== COMPREHENSIVE EVALUATION RESULTS ===")
            print(f"Test samples: {len(y_test)}")
            print(f"Actual cheaters: {y_test.sum()}")
            print(f"Predicted cheaters: {pred.sum()}")
            print(f"Ensemble Accuracy: {accuracy:.3f}")
            
            if len(np.unique(y_test)) > 1 and y_test.sum() > 0:
                try:
                    auc = roc_auc_score(y_test, prob)
                    print(f"AUC Score: {auc:.3f}")
                except:
                    auc = 0
                    print("Could not calculate AUC")
                
                # Confusion matrix
                cm = confusion_matrix(y_test, pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    print(f"Precision: {precision:.3f}")
                    print(f"Recall: {recall:.3f}")
                    print(f"False Positive Rate: {fpr:.3f}")
                    print(f"True Positives: {tp}")
                    print(f"False Positives: {fp}")
                    print(f"False Negatives: {fn}")
                    print(f"True Negatives: {tn}")
                else:
                    precision = recall = fpr = auc = 0
                    print("Confusion matrix shape unexpected")
            else:
                precision = recall = fpr = auc = 0
                print("Insufficient positive samples for detailed metrics")
            
            # Individual model comparison
            print(f"\n=== INDIVIDUAL MODEL PERFORMANCE ===")
            for name, pred_individual in individual_preds.items():
                acc = np.mean(pred_individual == y_test)
                print(f"{name}: {acc:.3f}")
                
            return {
                'ensemble_accuracy': accuracy,
                'auc_score': auc,
                'precision': precision,
                'recall': recall,
                'false_positive_rate': fpr
            }
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {'error': str(e)}
    
    def save_production_models(self, model_dir="production_models"):
        """Save all models for production deployment"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{model_dir}/{name}_model.joblib")
        
        # Save preprocessors
        if 'main' in self.scalers:
            joblib.dump(self.scalers['main'], f"{model_dir}/scaler.joblib")
        if self.feature_selector:
            joblib.dump(self.feature_selector, f"{model_dir}/feature_selector.joblib")
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'model_names': list(self.models.keys()),
            'trained_at': datetime.now().isoformat()
        }
        
        with open(f"{model_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Production models saved to {model_dir}/")

def main():
    """Train production anti-cheat system"""
    print("PRODUCTION ANTI-CHEAT ML SYSTEM")
    print("=" * 50)
    
    system = ProductionAntiCheatSystem()
    
    # Load dataset
    try:
        df = system.load_production_dataset()
        
        if len(df) < 100:
            print(f"WARNING: Dataset too small ({len(df)} players)")
            print("Need at least 100 players for meaningful training")
            print("Continuing with available data...")
        
        print(f"Dataset loaded: {len(df)} players")
        
        # Feature engineering
        X = system.engineer_advanced_features(df)
        y = system.create_ground_truth_labels(df)
        
        print(f"Features engineered: {X.shape[1]} features")
        print(f"Ground truth: {y.sum()} cheaters, {len(y) - y.sum()} clean")
        
        if y.sum() == 0:
            print("ERROR: No positive samples (cheaters) found!")
            print("Cannot train classification model without positive examples")
            return None, None
        
        # Train models
        X_test, y_test = system.train_ensemble_models(X, y)
        
        # Evaluation
        evaluation = system.comprehensive_evaluation(X_test, y_test)
        
        # Save models
        system.save_production_models()
        
        print(f"\nPRODUCTION SYSTEM STATUS:")
        if 'error' not in evaluation:
            print(f"Final ensemble accuracy: {evaluation['ensemble_accuracy']:.3f}")
            print(f"Models saved for deployment")
        else:
            print(f"Training completed with errors: {evaluation['error']}")
        
        return system, evaluation
        
    except Exception as e:
        print(f"Error in main training: {e}")
        return None, None

if __name__ == "__main__":
    main()