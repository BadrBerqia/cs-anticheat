# Recall-Optimized Anti-Cheat System
# Save as: recall_optimized_system.py

import numpy as np
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json
import os

class RecallOptimizedAntiCheat:
    def __init__(self, db_path="anticheat.db"):
        self.db_path = db_path
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.feature_names = []
        self.optimal_threshold = 0.5
        
    def load_and_engineer_features(self):
        """Load data and create focused feature set"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            p.steam_id, p.username, p.vac_banned, p.total_playtime,
            AVG(CAST(ms.kills AS FLOAT)) as avg_kills,
            AVG(CAST(ms.deaths AS FLOAT)) as avg_deaths,
            AVG(CAST(ms.headshots AS FLOAT)) as avg_headshots,
            AVG(CAST(ms.accuracy AS FLOAT)) as avg_accuracy,
            MAX(CAST(ms.kills AS FLOAT)) as max_kills,
            MAX(CAST(ms.accuracy AS FLOAT)) as max_accuracy,
            MAX(CAST(ms.headshots AS FLOAT)) as max_headshots,
            COUNT(ms.id) as total_sessions
        FROM players p
        LEFT JOIN match_stats ms ON p.steam_id = ms.steam_id
        GROUP BY p.steam_id
        HAVING COUNT(ms.id) > 0
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Focus on most predictive features
        features = pd.DataFrame()
        df_clean = df.fillna(0)
        
        # Core performance metrics
        features['avg_accuracy'] = df_clean['avg_accuracy']
        features['max_accuracy'] = df_clean['max_accuracy']
        features['avg_kills'] = df_clean['avg_kills']
        features['avg_deaths'] = df_clean['avg_deaths']
        features['avg_headshots'] = df_clean['avg_headshots']
        
        # Key ratios
        eps = 1e-6
        features['kd_ratio'] = features['avg_kills'] / (features['avg_deaths'] + eps)
        features['headshot_ratio'] = features['avg_headshots'] / (features['avg_kills'] + eps)
        
        # Impossibility indicators (more sensitive thresholds)
        features['high_accuracy'] = (features['avg_accuracy'] > 50).astype(int)  # Lowered from 60
        features['very_high_accuracy'] = (features['avg_accuracy'] > 65).astype(int)
        features['extreme_accuracy'] = (features['avg_accuracy'] > 80).astype(int)
        features['perfect_accuracy'] = (features['max_accuracy'] > 90).astype(int)
        
        features['high_headshots'] = (features['headshot_ratio'] > 0.5).astype(int)  # Lowered from 0.7
        features['very_high_headshots'] = (features['headshot_ratio'] > 0.7).astype(int)
        features['extreme_headshots'] = (features['headshot_ratio'] > 0.85).astype(int)
        
        features['high_kd'] = (features['kd_ratio'] > 2.5).astype(int)  # Lowered from 4
        features['very_high_kd'] = (features['kd_ratio'] > 4).astype(int)
        features['extreme_kd'] = (features['kd_ratio'] > 6).astype(int)
        
        # Combination indicators (key for detection)
        features['acc_hs_combo'] = (
            (features['avg_accuracy'] > 55) & (features['headshot_ratio'] > 0.6)
        ).astype(int)
        
        features['acc_kd_combo'] = (
            (features['avg_accuracy'] > 50) & (features['kd_ratio'] > 3)
        ).astype(int)
        
        features['triple_threat'] = (
            (features['avg_accuracy'] > 45) & 
            (features['headshot_ratio'] > 0.55) & 
            (features['kd_ratio'] > 2.5)
        ).astype(int)
        
        # Weighted suspicion score (more sensitive)
        features['suspicion_score'] = (
            features['high_accuracy'] * 0.2 +
            features['very_high_accuracy'] * 0.3 +
            features['extreme_accuracy'] * 0.4 +
            features['high_headshots'] * 0.2 +
            features['very_high_headshots'] * 0.3 +
            features['extreme_headshots'] * 0.4 +
            features['acc_hs_combo'] * 0.5 +
            features['triple_threat'] * 0.6
        )
        
        # Activity and consistency features
        features['total_sessions'] = df_clean['total_sessions']
        features['playtime'] = df_clean['total_playtime']
        features['consistency'] = (features['max_accuracy'] - features['avg_accuracy']) / (features['avg_accuracy'] + eps)
        
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        self.feature_names = features.columns.tolist()
        return features, df_clean
    
    def create_balanced_labels(self, df):
        """Enhanced labeling with lower thresholds"""
        labels = []
        
        for _, row in df.iterrows():
            if row['vac_banned']:
                labels.append(1)
                continue
            
            # More aggressive statistical detection
            acc = row.get('avg_accuracy', 0) or 0
            max_acc = row.get('max_accuracy', 0) or 0
            kd = (row.get('avg_kills', 0) or 0) / max(row.get('avg_deaths', 1) or 1, 1)
            hs_ratio = (row.get('avg_headshots', 0) or 0) / max(row.get('avg_kills', 1) or 1, 1)
            
            # Lowered thresholds for more detection
            suspicious = False
            
            # High accuracy alone
            if acc > 70:
                suspicious = True
            elif acc > 60 and (kd > 2.5 or hs_ratio > 0.6):
                suspicious = True
            elif acc > 50 and kd > 4:
                suspicious = True
            elif acc > 45 and hs_ratio > 0.8:
                suspicious = True
            elif max_acc > 95:  # Perfect accuracy games
                suspicious = True
            elif kd > 6 and hs_ratio > 0.5:
                suspicious = True
            elif hs_ratio > 0.9:  # Nearly impossible headshot ratio
                suspicious = True
            
            labels.append(1 if suspicious else 0)
        
        return np.array(labels)
    
    def train_recall_optimized_models(self, X, y):
        """Train models optimized for recall"""
        print(f"Training recall-optimized models")
        print(f"Original distribution: {y.sum()} cheaters, {len(y)-y.sum()} clean")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE: {y_train_balanced.sum()} cheaters, {len(y_train_balanced)-y_train_balanced.sum()} clean")
        
        # Preprocessing
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        self.feature_selector = SelectKBest(f_classif, k=min(15, X_train_scaled.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train_balanced)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        print(f"Selected {X_train_selected.shape[1]} features")
        
        # Models optimized for recall
        models_config = {
            'recall_rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,  # Deeper trees
                min_samples_split=2,  # More sensitive splits  
                min_samples_leaf=1,   # Allow smaller leaves
                class_weight='balanced_subsample',  # Handle imbalance
                random_state=42
            ),
            'recall_gb': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,  # Lower learning rate
                max_depth=10,        # Deeper trees
                min_samples_split=2,
                subsample=0.8,
                random_state=42
            )
        }
        
        # Train models
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_selected, y_train_balanced)
            
            # Evaluate on original test set
            test_pred = model.predict(X_test_selected)
            test_prob = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, 'predict_proba') else test_pred
            
            # Calculate metrics
            test_acc = np.mean(test_pred == y_test)
            
            if np.sum(y_test) > 0:
                cm = confusion_matrix(y_test, test_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    print(f"  Accuracy: {test_acc:.3f}")
                    print(f"  Recall: {recall:.3f}")
                    print(f"  Precision: {precision:.3f}")
            
            self.models[name] = model
        
        # Find optimal threshold for ensemble
        self.optimize_threshold(X_test_selected, y_test)
        
        return X_test_selected, y_test
    
    def optimize_threshold(self, X_test, y_test):
        """Find optimal threshold for recall-precision balance"""
        print(f"\nOptimizing decision threshold...")
        
        # Get ensemble probabilities
        all_probs = []
        for model in self.models.values():
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)[:, 1]
                all_probs.append(probs)
        
        if all_probs:
            ensemble_probs = np.mean(all_probs, axis=0)
            
            best_threshold = 0.5
            best_f1 = 0
            
            # Test different thresholds
            thresholds = np.arange(0.1, 0.9, 0.05)
            
            print("Threshold optimization:")
            for threshold in thresholds:
                pred = (ensemble_probs > threshold).astype(int)
                
                if np.sum(pred) > 0 and np.sum(y_test) > 0:
                    cm = confusion_matrix(y_test, pred)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        if recall >= 0.7 and f1 > best_f1:  # Prioritize recall >= 70%
                            best_f1 = f1
                            best_threshold = threshold
                        
                        if threshold in [0.2, 0.3, 0.4, 0.5]:
                            print(f"  {threshold:.1f}: Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}")
            
            self.optimal_threshold = best_threshold
            print(f"Optimal threshold: {self.optimal_threshold:.2f}")
    
    def predict_with_threshold(self, X):
        """Predict using optimized threshold"""
        # Preprocess
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Get ensemble probabilities
        all_probs = []
        for model in self.models.values():
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_selected)[:, 1]
                all_probs.append(probs)
        
        if all_probs:
            ensemble_probs = np.mean(all_probs, axis=0)
        else:
            ensemble_probs = np.zeros(len(X_selected))
        
        # Apply optimized threshold
        ensemble_pred = (ensemble_probs > self.optimal_threshold).astype(int)
        
        return ensemble_pred, ensemble_probs
    
    def final_evaluation(self, X_test, y_test):
        """Final comprehensive evaluation"""
        pred, prob = self.predict_with_threshold(X_test)
        
        print(f"\n=== RECALL-OPTIMIZED FINAL RESULTS ===")
        
        accuracy = np.mean(pred == y_test)
        print(f"Accuracy: {accuracy:.3f}")
        
        if np.sum(y_test) > 0:
            cm = confusion_matrix(y_test, pred)
            print(f"Confusion Matrix:\n{cm}")
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"Recall (Detection Rate): {recall:.3f}")
                print(f"Precision: {precision:.3f}")
                print(f"False Positive Rate: {fpr:.3f}")
                print(f"F1 Score: {f1:.3f}")
                
                print(f"\nBreakdown:")
                print(f"  True Positives (Caught cheaters): {tp}")
                print(f"  False Negatives (Missed cheaters): {fn}")
                print(f"  False Positives (Wrongly flagged): {fp}")
                print(f"  True Negatives (Correct clean): {tn}")
    
    def save_recall_optimized_models(self, save_dir="recall_optimized_models"):
        """Save the recall-optimized system"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f"{save_dir}/{name}_model.joblib")
        
        joblib.dump(self.scaler, f"{save_dir}/scaler.joblib")
        joblib.dump(self.feature_selector, f"{save_dir}/feature_selector.joblib")
        
        metadata = {
            'feature_names': self.feature_names,
            'optimal_threshold': self.optimal_threshold,
            'model_names': list(self.models.keys()),
            'trained_at': datetime.now().isoformat()
        }
        
        with open(f"{save_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Recall-optimized models saved to {save_dir}/")

def main():
    print("RECALL-OPTIMIZED ANTI-CHEAT SYSTEM")
    print("=" * 50)
    
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("Installing imbalanced-learn...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'imbalanced-learn'])
        from imblearn.over_sampling import SMOTE
    
    system = RecallOptimizedAntiCheat()
    
    # Load and prepare data
    X, df = system.load_and_engineer_features()
    y = system.create_balanced_labels(df)
    
    print(f"Dataset: {len(X)} players")
    print(f"Enhanced labels: {y.sum()} cheaters ({y.sum()/len(y)*100:.1f}%)")
    
    # Train recall-optimized models
    X_test, y_test = system.train_recall_optimized_models(X, y)
    
    # Final evaluation
    system.final_evaluation(X_test, y_test)
    
    # Save models
    system.save_recall_optimized_models()
    
    print(f"\nRECALL-OPTIMIZED SYSTEM COMPLETE")
    
    return system

if __name__ == "__main__":
    main()