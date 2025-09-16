# Comprehensive Evaluation and Visualization System
# Save as: comprehensive_evaluation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import sqlite3
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class ComprehensiveEvaluator:
    def __init__(self, model_dir="final_models", db_path="anticheat.db"):
        self.model_dir = model_dir
        self.db_path = db_path
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.metadata = None
        self.load_trained_system()
        
    def load_trained_system(self):
        """Load the trained models and preprocessors"""
        try:
            # Load metadata
            with open(f"{self.model_dir}/metadata.json", 'r') as f:
                self.metadata = json.load(f)
            
            # Load preprocessors
            self.scaler = joblib.load(f"{self.model_dir}/scaler.joblib")
            self.feature_selector = joblib.load(f"{self.model_dir}/feature_selector.joblib")
            
            # Load models
            for model_name in self.metadata['model_names']:
                self.models[model_name] = joblib.load(f"{self.model_dir}/{model_name}_model.joblib")
            
            print(f"Loaded system with {len(self.models)} models")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please run final_working_system.py first")
    
    def load_evaluation_data(self):
        """Load data for evaluation"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            p.steam_id, p.username, p.vac_banned,
            AVG(CAST(ms.kills AS FLOAT)) as avg_kills,
            AVG(CAST(ms.deaths AS FLOAT)) as avg_deaths,
            AVG(CAST(ms.headshots AS FLOAT)) as avg_headshots,
            AVG(CAST(ms.accuracy AS FLOAT)) as avg_accuracy,
            MAX(CAST(ms.kills AS FLOAT)) as max_kills,
            MAX(CAST(ms.accuracy AS FLOAT)) as max_accuracy,
            COUNT(ms.id) as total_sessions
        FROM players p
        LEFT JOIN match_stats ms ON p.steam_id = ms.steam_id
        GROUP BY p.steam_id
        HAVING COUNT(ms.id) > 0
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Apply same feature engineering as training
        features = self.engineer_features(df)
        labels = self.create_labels(df)
        
        return features, labels, df
    
    def engineer_features(self, df):
        """Apply same feature engineering as training"""
        df_clean = df.fillna(0)
        features = pd.DataFrame(index=df_clean.index)
        
        # Basic stats
        features['avg_kills'] = df_clean['avg_kills']
        features['avg_deaths'] = df_clean['avg_deaths'] 
        features['avg_accuracy'] = df_clean['avg_accuracy']
        features['avg_headshots'] = df_clean['avg_headshots']
        features['max_kills'] = df_clean['max_kills']
        features['max_accuracy'] = df_clean['max_accuracy']
        features['total_sessions'] = df_clean['total_sessions']
        
        # Derived features
        features['kd_ratio'] = features['avg_kills'] / np.maximum(features['avg_deaths'], 0.1)
        features['headshot_ratio'] = features['avg_headshots'] / np.maximum(features['avg_kills'], 0.1)
        features['accuracy_range'] = features['max_accuracy'] - features['avg_accuracy']
        
        # Binary indicators
        features['high_accuracy'] = (features['avg_accuracy'] > 50).astype(int)
        features['very_high_accuracy'] = (features['avg_accuracy'] > 65).astype(int)
        features['extreme_accuracy'] = (features['max_accuracy'] > 85).astype(int)
        features['high_headshots'] = (features['headshot_ratio'] > 0.5).astype(int)
        features['very_high_headshots'] = (features['headshot_ratio'] > 0.7).astype(int)
        features['high_kd'] = (features['kd_ratio'] > 2.5).astype(int)
        features['very_high_kd'] = (features['kd_ratio'] > 4).astype(int)
        
        # Combination features
        features['accuracy_headshot_combo'] = (
            (features['avg_accuracy'] > 55) & (features['headshot_ratio'] > 0.6)
        ).astype(int)
        features['accuracy_kd_combo'] = (
            (features['avg_accuracy'] > 50) & (features['kd_ratio'] > 3)
        ).astype(int)
        
        # Suspicion score
        features['suspicion_score'] = (
            features['high_accuracy'] * 0.2 +
            features['very_high_accuracy'] * 0.3 +
            features['extreme_accuracy'] * 0.4 +
            features['very_high_headshots'] * 0.3 +
            features['very_high_kd'] * 0.2 +
            features['accuracy_headshot_combo'] * 0.5 +
            features['accuracy_kd_combo'] * 0.4
        )
        
        return features.fillna(0).replace([np.inf, -np.inf], 0)
    
    def create_labels(self, df):
        """Apply same labeling logic as training"""
        labels = []
        for _, row in df.iterrows():
            if row['vac_banned']:
                labels.append(1)
            else:
                acc = row.get('avg_accuracy', 0) or 0
                max_acc = row.get('max_accuracy', 0) or 0
                kd = (row.get('avg_kills', 0) or 0) / max(row.get('avg_deaths', 1) or 1, 1)
                hs_ratio = (row.get('avg_headshots', 0) or 0) / max(row.get('avg_kills', 1) or 1, 1)
                
                if (acc > 70 or (acc > 60 and kd > 2.5) or (acc > 55 and hs_ratio > 0.7) or
                    (max_acc > 90) or (kd > 5 and hs_ratio > 0.5) or (hs_ratio > 0.85)):
                    labels.append(1)
                else:
                    labels.append(0)
        return np.array(labels)
    
    def cross_validation_analysis(self, X, y):
        """Perform cross-validation analysis"""
        print("Performing cross-validation analysis...")
        
        # Preprocess data
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X_selected, y, cv=skf, scoring='roc_auc')
            cv_results[name] = {
                'mean_auc': scores.mean(),
                'std_auc': scores.std(),
                'scores': scores
            }
            print(f"{name}: AUC = {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        return cv_results
    
    def create_performance_visualizations(self, X, y, output_dir="evaluation_plots"):
        """Create comprehensive performance visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess data
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Get predictions
        ensemble_probs = self.get_ensemble_predictions(X_selected)
        
        # 1. ROC Curves
        plt.figure(figsize=(12, 8))
        
        # Individual model ROC curves
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_probs = model.predict_proba(X_selected)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_probs)
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        # Ensemble ROC curve
        fpr, tpr, _ = roc_curve(y, ensemble_probs)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Ensemble (AUC = {auc_score:.3f})', 
                linewidth=3, linestyle='--', color='red')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Anti-Cheat Detection Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Precision-Recall Curves
        plt.figure(figsize=(12, 8))
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_probs = model.predict_proba(X_selected)[:, 1]
                precision, recall, _ = precision_recall_curve(y, y_probs)
                avg_precision = auc(recall, precision)
                plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})', linewidth=2)
        
        # Ensemble PR curve
        precision, recall, _ = precision_recall_curve(y, ensemble_probs)
        avg_precision = auc(recall, precision)
        plt.plot(recall, precision, label=f'Ensemble (AP = {avg_precision:.3f})', 
                linewidth=3, linestyle='--', color='red')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Anti-Cheat Detection Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Importance Analysis
        if 'rf_balanced' in self.models:
            rf_model = self.models['rf_balanced']
            feature_importance = rf_model.feature_importances_
            selected_features = self.metadata['selected_feature_names']
            
            # Sort features by importance
            importance_df = pd.DataFrame({
                'feature': selected_features,
                'importance': feature_importance
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(10, 8))
            bars = plt.barh(range(len(importance_df)), importance_df['importance'], 
                           color='skyblue', edgecolor='navy', alpha=0.7)
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title('Feature Importance - Random Forest Model', fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=10)
            
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Distribution Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Load original data for plotting
        _, _, df_original = self.load_evaluation_data()
        
        # Accuracy distribution
        axes[0,0].hist(df_original[df_original['vac_banned']==False]['avg_accuracy'], 
                      bins=30, alpha=0.7, label='Clean Players', color='green', density=True)
        axes[0,0].hist(df_original[df_original['vac_banned']==True]['avg_accuracy'], 
                      bins=30, alpha=0.7, label='VAC Banned', color='red', density=True)
        axes[0,0].set_xlabel('Average Accuracy (%)')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Accuracy Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # K/D Ratio distribution
        df_original['kd_ratio'] = df_original['avg_kills'] / np.maximum(df_original['avg_deaths'], 0.1)
        axes[0,1].hist(df_original[df_original['vac_banned']==False]['kd_ratio'], 
                      bins=30, alpha=0.7, label='Clean Players', color='green', density=True, range=(0, 10))
        axes[0,1].hist(df_original[df_original['vac_banned']==True]['kd_ratio'], 
                      bins=30, alpha=0.7, label='VAC Banned', color='red', density=True, range=(0, 10))
        axes[0,1].set_xlabel('K/D Ratio')
        axes[0,1].set_ylabel('Density')
        axes[0,1].set_title('K/D Ratio Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Headshot ratio distribution
        df_original['headshot_ratio'] = df_original['avg_headshots'] / np.maximum(df_original['avg_kills'], 0.1)
        axes[1,0].hist(df_original[df_original['vac_banned']==False]['headshot_ratio'], 
                      bins=30, alpha=0.7, label='Clean Players', color='green', density=True, range=(0, 1))
        axes[1,0].hist(df_original[df_original['vac_banned']==True]['headshot_ratio'], 
                      bins=30, alpha=0.7, label='VAC Banned', color='red', density=True, range=(0, 1))
        axes[1,0].set_xlabel('Headshot Ratio')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Headshot Ratio Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Prediction confidence distribution
        axes[1,1].hist(ensemble_probs[y==0], bins=30, alpha=0.7, label='Clean Players', 
                      color='green', density=True)
        axes[1,1].hist(ensemble_probs[y==1], bins=30, alpha=0.7, label='Cheaters', 
                      color='red', density=True)
        axes[1,1].axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Threshold')
        axes[1,1].set_xlabel('Model Confidence Score')
        axes[1,1].set_ylabel('Density')
        axes[1,1].set_title('Model Confidence Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
    
    def get_ensemble_predictions(self, X_processed):
        """Get ensemble predictions from preprocessed data"""
        all_probs = []
        for model in self.models.values():
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_processed)[:, 1]
                all_probs.append(probs)
        
        return np.mean(all_probs, axis=0) if all_probs else np.zeros(len(X_processed))
    
    def business_impact_analysis(self, X, y, player_data):
        """Analyze business impact of different threshold choices"""
        print("\nBusiness Impact Analysis")
        print("=" * 40)
        
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        ensemble_probs = self.get_ensemble_predictions(X_selected)
        
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []
        
        for threshold in thresholds:
            pred = (ensemble_probs > threshold).astype(int)
            
            # Calculate metrics
            tp = np.sum((pred == 1) & (y == 1))
            fp = np.sum((pred == 1) & (y == 0))
            fn = np.sum((pred == 0) & (y == 1))
            tn = np.sum((pred == 0) & (y == 0))
            
            # Business metrics
            total_flagged = tp + fp
            total_cheaters = tp + fn
            total_clean = fp + tn
            
            recall = tp / total_cheaters if total_cheaters > 0 else 0
            precision = tp / total_flagged if total_flagged > 0 else 0
            fpr = fp / total_clean if total_clean > 0 else 0
            
            results.append({
                'threshold': threshold,
                'flagged_players': total_flagged,
                'caught_cheaters': tp,
                'false_accusations': fp,
                'missed_cheaters': fn,
                'recall': recall,
                'precision': precision,
                'fpr': fpr
            })
        
        # Display business impact table
        df_results = pd.DataFrame(results)
        print("\nThreshold Impact Analysis:")
        print("Threshold | Flagged | Caught | False+ | Missed | Recall | Precision | FPR")
        print("-" * 75)
        for _, row in df_results.iterrows():
            print(f"{row['threshold']:8.1f} | {int(row['flagged_players']):7d} | "
                  f"{int(row['caught_cheaters']):6d} | {int(row['false_accusations']):6d} | "
                  f"{int(row['missed_cheaters']):6d} | {row['recall']:6.3f} | "
                  f"{row['precision']:9.3f} | {row['fpr']:7.3f}")
        
        return df_results
    
    def generate_final_report(self, output_file="evaluation_report.txt"):
        """Generate comprehensive evaluation report"""
        X, y, df = self.load_evaluation_data()
        
        report = []
        report.append("COMPREHENSIVE ANTI-CHEAT SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Dataset overview
        report.append("DATASET OVERVIEW")
        report.append("-" * 20)
        report.append(f"Total players analyzed: {len(df)}")
        report.append(f"VAC banned players: {df['vac_banned'].sum()}")
        report.append(f"Statistical suspects: {y.sum() - df['vac_banned'].sum()}")
        report.append(f"Clean players: {len(df) - y.sum()}")
        report.append(f"Cheater prevalence: {y.sum()/len(y)*100:.1f}%")
        report.append("")
        
        # Model performance
        cv_results = self.cross_validation_analysis(X, y)
        report.append("CROSS-VALIDATION RESULTS")
        report.append("-" * 25)
        for name, results in cv_results.items():
            report.append(f"{name}: AUC = {results['mean_auc']:.3f} (+/- {results['std_auc']*2:.3f})")
        report.append("")
        
        # Business impact
        business_results = self.business_impact_analysis(X, y, df)
        report.append("RECOMMENDED OPERATING POINTS")
        report.append("-" * 30)
        
        # Find conservative threshold (high precision)
        conservative = business_results[business_results['precision'] >= 0.4]
        if len(conservative) > 0:
            best_conservative = conservative.loc[conservative['recall'].idxmax()]
            report.append(f"Conservative (High Precision): Threshold {best_conservative['threshold']:.1f}")
            report.append(f"  - Flags {best_conservative['flagged_players']} players, catches {best_conservative['caught_cheaters']} cheaters")
            report.append(f"  - Precision: {best_conservative['precision']:.3f}, Recall: {best_conservative['recall']:.3f}")
        
        # Find balanced threshold
        business_results['f1'] = 2 * (business_results['precision'] * business_results['recall']) / (business_results['precision'] + business_results['recall'])
        best_balanced = business_results.loc[business_results['f1'].idxmax()]
        report.append(f"Balanced: Threshold {best_balanced['threshold']:.1f}")
        report.append(f"  - Flags {best_balanced['flagged_players']} players, catches {best_balanced['caught_cheaters']} cheaters")
        report.append(f"  - Precision: {best_balanced['precision']:.3f}, Recall: {best_balanced['recall']:.3f}")
        
        report.append("")
        report.append("SYSTEM LIMITATIONS AND RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. High false positive rate requires human review for flagged players")
        report.append("2. System best suited for initial screening, not automated banning")
        report.append("3. Performance limited by statistical analysis approach")
        report.append("4. Consider temporal behavior analysis for future improvements")
        
        # Save report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Comprehensive report saved to {output_file}")
        return '\n'.join(report)

def main():
    print("COMPREHENSIVE ANTI-CHEAT SYSTEM EVALUATION")
    print("=" * 50)
    
    evaluator = ComprehensiveEvaluator()
    
    if not evaluator.models:
        print("No trained models found. Please run final_working_system.py first.")
        return
    
    # Load evaluation data
    X, y, df = evaluator.load_evaluation_data()
    print(f"Evaluation dataset: {len(X)} players, {y.sum()} cheaters")
    
    # Perform comprehensive evaluation
    print("\n1. Cross-validation analysis...")
    cv_results = evaluator.cross_validation_analysis(X, y)
    
    print("\n2. Creating performance visualizations...")
    evaluator.create_performance_visualizations(X, y)
    
    print("\n3. Business impact analysis...")
    business_results = evaluator.business_impact_analysis(X, y, df)
    
    print("\n4. Generating final report...")
    report = evaluator.generate_final_report()
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("Files generated:")
    print("- evaluation_plots/ (ROC curves, feature importance, distributions)")
    print("- evaluation_report.txt (comprehensive analysis)")
    print("\nRecommendation: Use threshold 0.6-0.7 for human-review system")
    print("Lower thresholds catch more cheaters but create too many false positives")

if __name__ == "__main__":
    main()