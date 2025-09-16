# Advanced Feature Engineering for Anti-Cheat System
# Save as: advanced_features.py

import numpy as np
import pandas as pd
import sqlite3
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
import json
import os
from datetime import datetime

class AdvancedAntiCheatSystem:
    def __init__(self, db_path="anticheat.db"):
        self.db_path = db_path
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.feature_names = []
        
    def load_enhanced_data(self):
        """Load data with temporal analysis"""
        conn = sqlite3.connect(self.db_path)
        
        # Enhanced query with more statistical features
        query = """
        SELECT 
            p.steam_id,
            p.username,
            p.vac_banned,
            p.total_playtime,
            
            -- Basic aggregations
            AVG(CAST(ms.kills AS FLOAT)) as avg_kills,
            AVG(CAST(ms.deaths AS FLOAT)) as avg_deaths,
            AVG(CAST(ms.headshots AS FLOAT)) as avg_headshots,
            AVG(CAST(ms.accuracy AS FLOAT)) as avg_accuracy,
            
            -- Statistical measures
            MAX(CAST(ms.kills AS FLOAT)) as max_kills,
            MIN(CAST(ms.kills AS FLOAT)) as min_kills,
            MAX(CAST(ms.accuracy AS FLOAT)) as max_accuracy,
            MIN(CAST(ms.accuracy AS FLOAT)) as min_accuracy,
            MAX(CAST(ms.headshots AS FLOAT)) as max_headshots,
            
            -- Count features
            COUNT(ms.id) as total_sessions,
            
            -- Variance approximation (using available SQL functions)
            (MAX(CAST(ms.kills AS FLOAT)) - MIN(CAST(ms.kills AS FLOAT))) as kill_range,
            (MAX(CAST(ms.accuracy AS FLOAT)) - MIN(CAST(ms.accuracy AS FLOAT))) as accuracy_range
            
        FROM players p
        LEFT JOIN match_stats ms ON p.steam_id = ms.steam_id
        GROUP BY p.steam_id
        HAVING COUNT(ms.id) > 0
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def engineer_advanced_features(self, df):
        """Create sophisticated feature set"""
        features = pd.DataFrame()
        df_clean = df.fillna(0)
        
        # === PERFORMANCE FEATURES ===
        features['avg_kills'] = df_clean['avg_kills']
        features['avg_deaths'] = df_clean['avg_deaths']
        features['avg_accuracy'] = df_clean['avg_accuracy']
        features['avg_headshots'] = df_clean['avg_headshots']
        features['total_playtime'] = df_clean['total_playtime']
        features['total_sessions'] = df_clean['total_sessions']
        
        # === DERIVED METRICS ===
        # Safe division with small epsilon
        eps = 1e-6
        features['kd_ratio'] = features['avg_kills'] / (features['avg_deaths'] + eps)
        features['headshot_ratio'] = features['avg_headshots'] / (features['avg_kills'] + eps)
        features['playtime_per_session'] = features['total_playtime'] / (features['total_sessions'] + eps)
        
        # === CONSISTENCY FEATURES ===
        features['kill_range'] = df_clean['kill_range']
        features['accuracy_range'] = df_clean['accuracy_range']
        features['kill_consistency'] = features['kill_range'] / (features['avg_kills'] + eps)
        features['accuracy_consistency'] = features['accuracy_range'] / (features['avg_accuracy'] + eps)
        
        # === EXTREME PERFORMANCE INDICATORS ===
        features['max_kills'] = df_clean['max_kills']
        features['max_accuracy'] = df_clean['max_accuracy']
        features['max_headshots'] = df_clean['max_headshots']
        
        # Binary indicators for extreme values
        features['extreme_accuracy'] = (features['max_accuracy'] > 90).astype(int)
        features['impossible_accuracy'] = (features['avg_accuracy'] > 70).astype(int)
        features['extreme_headshots'] = (features['headshot_ratio'] > 0.8).astype(int)
        features['extreme_kd'] = (features['kd_ratio'] > 5).astype(int)
        features['perfect_sessions'] = (features['max_accuracy'] > 95).astype(int)
        
        # === STATISTICAL IMPOSSIBILITY SCORES ===
        # Weighted impossibility score
        features['impossibility_score'] = (
            features['impossible_accuracy'] * 0.4 +
            features['extreme_headshots'] * 0.3 +
            features['extreme_kd'] * 0.2 +
            features['perfect_sessions'] * 0.1
        )
        
        # === BEHAVIORAL PATTERNS ===
        # Consistency vs performance (cheaters often have low variance + high performance)
        features['suspicious_consistency'] = (
            (features['accuracy_consistency'] < 0.3) & (features['avg_accuracy'] > 50)
        ).astype(int)
        
        # Activity patterns
        features['session_intensity'] = features['total_sessions'] / (features['total_playtime'] / 60 + eps)
        features['high_activity'] = (features['total_sessions'] > 20).astype(int)
        
        # === MATHEMATICAL COMBINATIONS ===
        # Multiplicative features (interaction terms)
        features['kd_accuracy_product'] = features['kd_ratio'] * features['avg_accuracy']
        features['headshot_accuracy_product'] = features['headshot_ratio'] * features['avg_accuracy']
        
        # Ratios and normalized features
        features['kills_per_session'] = features['avg_kills'] * features['total_sessions']
        features['headshots_per_session'] = features['avg_headshots'] * features['total_sessions']
        
        # === PERCENTILE-BASED FEATURES ===
        # Create percentile ranks for key metrics (helps with outlier detection)
        for col in ['avg_accuracy', 'kd_ratio', 'headshot_ratio']:
            if col in features.columns:
                features[f'{col}_percentile'] = features[col].rank(pct=True)
        
        # === OUTLIER DETECTION FEATURES ===
        # Z-score based outlier detection
        for col in ['avg_accuracy', 'kd_ratio', 'headshot_ratio']:
            if col in features.columns:
                mean_val = features[col].mean()
                std_val = features[col].std()
                features[f'{col}_zscore'] = np.abs((features[col] - mean_val) / (std_val + eps))
                features[f'{col}_is_outlier'] = (features[f'{col}_zscore'] > 2.5).astype(int)
        
        # === COMPOSITE SCORES ===
        # Overall skill score (normalized)
        skill_components = ['kd_ratio', 'avg_accuracy', 'headshot_ratio']
        skill_scores = []
        for component in skill_components:
            if component in features.columns:
                # Normalize to 0-1 range
                col_data = features[component]
                normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min() + eps)
                skill_scores.append(normalized)
        
        if skill_scores:
            features['overall_skill'] = np.mean(skill_scores, axis=0)
        
        # Final cleanup
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        self.feature_names = features.columns.tolist()
        print(f"Engineered {len(self.feature_names)} advanced features")
        
        return features
    
    def create_enhanced_labels(self, df):
        """Enhanced ground truth with multiple criteria"""
        labels = []
        
        for _, row in df.iterrows():
            # Primary ground truth: VAC bans
            if row['vac_banned']:
                labels.append(1)
                continue
            
            # Enhanced statistical impossibility detection
            acc = row.get('avg_accuracy', 0) or 0
            max_acc = row.get('max_accuracy', 0) or 0
            kd = (row.get('avg_kills', 0) or 0) / max(row.get('avg_deaths', 1) or 1, 1)
            hs_ratio = (row.get('avg_headshots', 0) or 0) / max(row.get('avg_kills', 1) or 1, 1)
            
            # Multiple impossibility criteria
            impossibility_flags = 0
            
            if acc > 75:  # Extremely high average accuracy
                impossibility_flags += 3
            elif acc > 60:  # High accuracy
                impossibility_flags += 2
            elif acc > 50:  # Moderately high accuracy
                impossibility_flags += 1
                
            if max_acc > 95:  # Perfect accuracy sessions
                impossibility_flags += 2
            elif max_acc > 85:  # Near-perfect sessions
                impossibility_flags += 1
                
            if kd > 6:  # Extremely high K/D
                impossibility_flags += 2
            elif kd > 4:  # High K/D
                impossibility_flags += 1
                
            if hs_ratio > 0.9:  # Near-impossible headshot ratio
                impossibility_flags += 3
            elif hs_ratio > 0.7:  # Very high headshot ratio
                impossibility_flags += 2
            elif hs_ratio > 0.6:  # High headshot ratio
                impossibility_flags += 1
            
            # Combined criteria for positive classification
            if impossibility_flags >= 4:  # Multiple strong indicators
                labels.append(1)
            elif impossibility_flags >= 6:  # Very strong evidence
                labels.append(1)
            else:
                labels.append(0)
        
        return np.array(labels)
    
    def train_advanced_models(self, X, y):
        """Train with advanced techniques"""
        print(f"Training advanced models on {len(X)} samples")
        print(f"Feature dimensions: {X.shape}")
        print(f"Class distribution: {y.sum()} cheaters, {len(y)-y.sum()} clean")
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Advanced preprocessing
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection with multiple methods
        # Method 1: Statistical selection
        selector_stats = SelectKBest(f_classif, k=20)
        X_train_stats = selector_stats.fit_transform(X_train_scaled, y_train)
        
        # Method 2: Recursive feature elimination with RF
        rf_selector = RandomForestClassifier(n_estimators=50, random_state=42)
        selector_rfe = RFE(rf_selector, n_features_to_select=15)
        X_train_rfe = selector_rfe.fit_transform(X_train_scaled, y_train)
        
        # Use statistical selection as primary method
        self.feature_selector = selector_stats
        X_train_final = X_train_stats
        X_test_final = self.feature_selector.transform(X_test_scaled)
        
        print(f"Selected {X_train_final.shape[1]} features from {X_train_scaled.shape[1]}")
        
        # Train ensemble of models
        models_config = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=5,
                random_state=42
            ),
        }
        
        results = {}
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_final, y_train)
            
            train_acc = model.score(X_train_final, y_train)
            test_acc = model.score(X_test_final, y_test)
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                selected_features = np.array(self.feature_names)[self.feature_selector.get_support()]
                feature_importance = list(zip(selected_features, importance))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                print(f"  Top 5 features for {name}:")
                for feat, imp in feature_importance[:5]:
                    print(f"    {feat}: {imp:.3f}")
            
            print(f"  Accuracy: Train {train_acc:.3f}, Test {test_acc:.3f}")
            
            self.models[name] = model
            results[name] = test_acc
        
        # Detailed evaluation
        self.evaluate_advanced_models(X_test_final, y_test)
        
        return results, (X_test_final, y_test)
    
    def evaluate_advanced_models(self, X_test, y_test):
        """Comprehensive evaluation with visualizations"""
        ensemble_pred, ensemble_prob, individual_preds = self.predict_preprocessed(X_test)
        
        print(f"\n=== ADVANCED EVALUATION RESULTS ===")
        
        # Basic metrics
        accuracy = np.mean(ensemble_pred == y_test)
        print(f"Ensemble Accuracy: {accuracy:.3f}")
        
        if len(np.unique(y_test)) > 1 and np.sum(y_test) > 0:
            auc = roc_auc_score(y_test, ensemble_prob)
            print(f"AUC Score: {auc:.3f}")
            
            # Detailed classification report
            from sklearn.metrics import classification_report
            print("\nClassification Report:")
            print(classification_report(y_test, ensemble_pred, target_names=['Clean', 'Cheater']))
            
            # ROC Curve data (for plotting later)
            fpr, tpr, thresholds = roc_curve(y_test, ensemble_prob)
            
            # Find optimal threshold
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            print(f"Optimal threshold: {optimal_threshold:.3f}")
            
            # Performance at different thresholds
            print(f"\nThreshold Analysis:")
            for threshold in [0.3, 0.5, 0.7, 0.9]:
                pred_at_threshold = (ensemble_prob > threshold).astype(int)
                acc_at_threshold = np.mean(pred_at_threshold == y_test)
                if np.sum(pred_at_threshold) > 0:
                    precision_at_threshold = np.sum((pred_at_threshold == 1) & (y_test == 1)) / np.sum(pred_at_threshold)
                else:
                    precision_at_threshold = 0
                recall_at_threshold = np.sum((pred_at_threshold == 1) & (y_test == 1)) / np.sum(y_test) if np.sum(y_test) > 0 else 0
                
                print(f"  Threshold {threshold}: Acc={acc_at_threshold:.3f}, Prec={precision_at_threshold:.3f}, Recall={recall_at_threshold:.3f}")
    
    def predict_preprocessed(self, X_preprocessed):
        """Predict on preprocessed data"""
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
        
        # Ensemble: weighted average
        weights = {'random_forest': 0.6, 'gradient_boost': 0.4}
        ensemble_prob = sum(weights.get(name, 0.5) * prob for name, prob in probabilities.items())
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        return ensemble_pred, ensemble_prob, predictions
    
    def save_advanced_models(self, save_dir="advanced_models"):
        """Save the advanced model system"""
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
            'selected_features': len(self.feature_selector.get_support()) if self.feature_selector else 0,
            'model_names': list(self.models.keys()),
            'trained_at': datetime.now().isoformat()
        }
        
        with open(f"{save_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Advanced models saved to {save_dir}/")

def main():
    print("ADVANCED ANTI-CHEAT SYSTEM WITH ENHANCED FEATURES")
    print("=" * 60)
    
    system = AdvancedAntiCheatSystem()
    
    # Load enhanced data
    df = system.load_enhanced_data()
    print(f"Loaded: {len(df)} players")
    
    # Engineer advanced features
    X = system.engineer_advanced_features(df)
    y = system.create_enhanced_labels(df)
    
    print(f"Enhanced labels: {y.sum()} cheaters out of {len(y)} players")
    
    # Train advanced models
    results, test_data = system.train_advanced_models(X, y)
    
    # Save models
    system.save_advanced_models()
    
    print(f"\nADVANCED SYSTEM COMPLETE")
    print("Enhanced models ready for deployment")
    
    return system

if __name__ == "__main__":
    main()