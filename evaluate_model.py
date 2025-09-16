# Anti-Cheat Model Evaluation and Accuracy Testing
# Save as: evaluate_model.py

import sys
import os
sys.path.append('src')

import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from utils.database import AntiCheatDatabase

class AntiCheatEvaluator:
    def __init__(self):
        self.db = AntiCheatDatabase()
        
    def load_evaluation_data(self):
        """Load data with ground truth labels"""
        conn = self.db.get_connection()
        
        # Get all players with their match stats and create ground truth
        query = '''
            SELECT p.steam_id, p.username, p.vac_banned,
                   AVG(ms.kills) as avg_kills, AVG(ms.deaths) as avg_deaths,
                   AVG(ms.headshots) as avg_headshots, AVG(ms.accuracy) as avg_accuracy,
                   COUNT(ms.id) as total_matches,
                   MAX(ms.accuracy) as max_accuracy
            FROM players p
            LEFT JOIN match_stats ms ON p.steam_id = ms.steam_id
            GROUP BY p.steam_id
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Fill missing values
        df = df.fillna(0)
        
        # Create ground truth labels
        # 1 = Cheater (VAC banned or extremely suspicious stats)
        # 0 = Clean player
        df['ground_truth'] = df.apply(self._create_ground_truth_label, axis=1)
        
        print(f"📊 Evaluation dataset: {len(df)} players")
        print(f"   Cheaters (ground truth): {df['ground_truth'].sum()}")
        print(f"   Clean players: {len(df) - df['ground_truth'].sum()}")
        
        return df
    
    def _create_ground_truth_label(self, row):
        """Create ground truth labels based on known information"""
        # VAC banned = definitely cheating
        if row['vac_banned']:
            return 1
            
        # Statistical impossibilities = likely cheating
        kd_ratio = row['avg_kills'] / max(row['avg_deaths'], 1)
        headshot_ratio = row['avg_headshots'] / max(row['avg_kills'], 1)
        
        # Very suspicious combinations
        if (row['avg_accuracy'] > 70 and headshot_ratio > 0.8) or \
           (row['avg_accuracy'] > 75) or \
           (kd_ratio > 6 and headshot_ratio > 0.7):
            return 1
            
        return 0
    
    def predict_with_current_model(self, df):
        """Make predictions using current statistical model"""
        predictions = []
        confidence_scores = []
        
        for _, row in df.iterrows():
            # Same logic as in run_anticheat.py
            anomaly_score = 0.0
            
            # VAC ban check
            if row['vac_banned']:
                anomaly_score = 0.95
            else:
                # Statistical analysis
                kd_ratio = row['avg_kills'] / max(row['avg_deaths'], 1)
                headshot_ratio = row['avg_headshots'] / max(row['avg_kills'], 1)
                
                if row['avg_accuracy'] > 60:
                    anomaly_score += 0.4
                if headshot_ratio > 0.7:
                    anomaly_score += 0.3
                if kd_ratio > 4:
                    anomaly_score += 0.2
                if row['avg_accuracy'] > 75:
                    anomaly_score += 0.3
            
            confidence_scores.append(anomaly_score)
            predictions.append(1 if anomaly_score > 0.5 else 0)
        
        return np.array(predictions), np.array(confidence_scores)
    
    def calculate_metrics(self, y_true, y_pred, y_scores):
        """Calculate comprehensive evaluation metrics"""
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # False positive rate (critical for anti-cheat)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # AUC score
        try:
            auc_score = roc_auc_score(y_true, y_scores)
        except:
            auc_score = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'false_positive_rate': fpr,
            'auc_score': auc_score,
            'confusion_matrix': {
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp
            }
        }
    
    def analyze_false_positives(self, df, y_true, y_pred):
        """Analyze false positive cases"""
        false_positives = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            if y_true[i] == 0 and y_pred[i] == 1:  # Predicted cheater but actually clean
                false_positives.append({
                    'username': row['username'],
                    'steam_id': row['steam_id'],
                    'avg_accuracy': row['avg_accuracy'],
                    'avg_kills': row['avg_kills'],
                    'avg_deaths': row['avg_deaths'],
                    'kd_ratio': row['avg_kills'] / max(row['avg_deaths'], 1),
                    'headshot_ratio': row['avg_headshots'] / max(row['avg_kills'], 1)
                })
        
        return false_positives
    
    def analyze_false_negatives(self, df, y_true, y_pred):
        """Analyze false negative cases"""
        false_negatives = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            if y_true[i] == 1 and y_pred[i] == 0:  # Predicted clean but actually cheater
                false_negatives.append({
                    'username': row['username'],
                    'steam_id': row['steam_id'],
                    'vac_banned': row['vac_banned'],
                    'avg_accuracy': row['avg_accuracy'],
                    'avg_kills': row['avg_kills'],
                    'avg_deaths': row['avg_deaths'],
                    'kd_ratio': row['avg_kills'] / max(row['avg_deaths'], 1),
                    'headshot_ratio': row['avg_headshots'] / max(row['avg_kills'], 1)
                })
        
        return false_negatives
    
    def test_different_thresholds(self, df, y_true, y_scores):
        """Test different detection thresholds"""
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        results = []
        
        for threshold in thresholds:
            y_pred = (y_scores > threshold).astype(int)
            metrics = self.calculate_metrics(y_true, y_pred, y_scores)
            
            results.append({
                'threshold': threshold,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'fpr': metrics['false_positive_rate'],
                'f1_score': metrics['f1_score']
            })
        
        return results
    
    def benchmark_against_simple_rules(self, df):
        """Compare against simple rule-based detection"""
        # Simple rule: VAC banned OR accuracy > 70%
        simple_predictions = ((df['vac_banned'] == 1) | (df['avg_accuracy'] > 70)).astype(int)
        
        # Even simpler: Just VAC bans
        vac_only_predictions = (df['vac_banned'] == 1).astype(int)
        
        return simple_predictions, vac_only_predictions
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        print("🔍 ANTI-CHEAT MODEL EVALUATION REPORT")
        print("=" * 60)
        
        # Load data
        df = self.load_evaluation_data()
        
        if len(df) == 0:
            print("❌ No data available for evaluation!")
            return
        
        # Get ground truth
        y_true = df['ground_truth'].values
        
        # Get predictions from current model
        y_pred, y_scores = self.predict_with_current_model(df)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_scores)
        
        # Print main results
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Accuracy: {metrics['accuracy']:.2%}")
        print(f"   Precision: {metrics['precision']:.2%} (of flagged players, how many are actually cheaters)")
        print(f"   Recall: {metrics['recall']:.2%} (of actual cheaters, how many did we catch)")
        print(f"   F1-Score: {metrics['f1_score']:.3f}")
        print(f"   AUC Score: {metrics['auc_score']:.3f}")
        
        # Critical metrics for anti-cheat
        print(f"\n🎯 CRITICAL ANTI-CHEAT METRICS:")
        print(f"   False Positive Rate: {metrics['false_positive_rate']:.2%} (clean players wrongly banned)")
        print(f"   False Negative Rate: {1-metrics['recall']:.2%} (cheaters not detected)")
        print(f"   Specificity: {metrics['specificity']:.2%} (clean players correctly identified)")
        
        # Confusion Matrix
        cm = metrics['confusion_matrix']
        print(f"\n📈 CONFUSION MATRIX:")
        print(f"   True Negatives (Clean → Clean): {cm['true_negatives']}")
        print(f"   False Positives (Clean → Cheater): {cm['false_positives']} ⚠️")
        print(f"   False Negatives (Cheater → Clean): {cm['false_negatives']} ⚠️")
        print(f"   True Positives (Cheater → Cheater): {cm['true_positives']}")
        
        # Analyze errors
        false_positives = self.analyze_false_positives(df, y_true, y_pred)
        false_negatives = self.analyze_false_negatives(df, y_true, y_pred)
        
        if false_positives:
            print(f"\n🚨 FALSE POSITIVES ({len(false_positives)} cases):")
            print("   Clean players wrongly flagged as cheaters:")
            for i, fp in enumerate(false_positives[:5], 1):
                print(f"   {i}. {fp['username']}: Acc={fp['avg_accuracy']:.1f}%, K/D={fp['kd_ratio']:.2f}")
        
        if false_negatives:
            print(f"\n⚠️  FALSE NEGATIVES ({len(false_negatives)} cases):")
            print("   Cheaters that weren't detected:")
            for i, fn in enumerate(false_negatives[:5], 1):
                vac_status = "VAC BANNED" if fn['vac_banned'] else "Suspicious Stats"
                print(f"   {i}. {fn['username']} [{vac_status}]: Acc={fn['avg_accuracy']:.1f}%, K/D={fn['kd_ratio']:.2f}")
        
        # Test different thresholds
        print(f"\n🎚️  THRESHOLD ANALYSIS:")
        threshold_results = self.test_different_thresholds(df, y_true, y_scores)
        print("   Threshold | Accuracy | Precision | Recall | FPR    | F1")
        print("   ----------|----------|-----------|--------|--------|-------")
        for result in threshold_results:
            print(f"   {result['threshold']:.1f}       | {result['accuracy']:.2%}    | {result['precision']:.2%}     | {result['recall']:.2%}  | {result['fpr']:.2%} | {result['f1_score']:.3f}")
        
        # Benchmark comparison
        simple_pred, vac_pred = self.benchmark_against_simple_rules(df)
        simple_metrics = self.calculate_metrics(y_true, simple_pred, simple_pred.astype(float))
        vac_metrics = self.calculate_metrics(y_true, vac_pred, vac_pred.astype(float))
        
        print(f"\n🏆 BENCHMARK COMPARISON:")
        print(f"   Method              | Accuracy | Precision | Recall | FPR")
        print(f"   --------------------|----------|-----------|--------|--------")
        print(f"   Current Model       | {metrics['accuracy']:.2%}    | {metrics['precision']:.2%}     | {metrics['recall']:.2%}  | {metrics['false_positive_rate']:.2%}")
        print(f"   Simple Rules        | {simple_metrics['accuracy']:.2%}    | {simple_metrics['precision']:.2%}     | {simple_metrics['recall']:.2%}  | {simple_metrics['false_positive_rate']:.2%}")
        print(f"   VAC Bans Only       | {vac_metrics['accuracy']:.2%}    | {vac_metrics['precision']:.2%}     | {vac_metrics['recall']:.2%}  | {vac_metrics['false_positive_rate']:.2%}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if metrics['false_positive_rate'] > 0.05:
            print("   ⚠️  High false positive rate - consider stricter thresholds")
        if metrics['recall'] < 0.8:
            print("   ⚠️  Low recall - missing too many cheaters, consider looser thresholds")
        if metrics['precision'] < 0.9:
            print("   ⚠️  Low precision - too many false accusations")
        
        # Optimal threshold recommendation
        best_threshold = max(threshold_results, key=lambda x: x['f1_score'])
        print(f"   🎯 Optimal threshold for F1-Score: {best_threshold['threshold']}")
        
        # Data quality assessment
        print(f"\n📊 DATA QUALITY ASSESSMENT:")
        total_cheaters = y_true.sum()
        total_clean = len(y_true) - total_cheaters
        
        if total_cheaters < 5:
            print("   ⚠️  Very few confirmed cheaters - collect more VAC banned players")
        if total_clean < 20:
            print("   ⚠️  Need more clean player data for reliable evaluation")
        if len(df) < 50:
            print("   ⚠️  Small dataset - collect more players for robust evaluation")
        else:
            print("   ✅ Sufficient data for evaluation")
        
        return metrics

def main():
    """Run model evaluation"""
    evaluator = AntiCheatEvaluator()
    evaluator.generate_evaluation_report()

if __name__ == "__main__":
    main()