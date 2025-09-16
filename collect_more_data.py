# Quick script to collect more training data
# Save as: collect_more_data.py

import sys
import os
sys.path.append('src')

from data.steam_collector import SteamDataCollector
from dotenv import load_dotenv

def collect_training_data():
    """Collect more players for better ML training"""
    load_dotenv()
    api_key = os.getenv('STEAM_API_KEY')
    
    if not api_key:
        print("❌ No Steam API key found!")
        return
    
    collector = SteamDataCollector(api_key)
    
    print("🎯 Collecting larger dataset for ML training...")
    print("This will take a few minutes to find players with game statistics...")
    
    # Collect more players
    dataset = collector.collect_dataset(target_count=100)
    
    print(f"\n✅ Collection complete!")
    print(f"📊 Total players collected: {len(dataset)}")
    
    # Analyze the dataset
    vac_banned = [p for p in dataset if p['vac_banned']]
    high_anomaly = [p for p in dataset if p['anomaly_score'] > 0.5]
    
    print(f"🚨 VAC banned players: {len(vac_banned)}")
    print(f"⚠️  High anomaly score players: {len(high_anomaly)}")
    
    if high_anomaly:
        print("\n🔍 Most suspicious players found:")
        sorted_suspicious = sorted(high_anomaly, key=lambda x: x['anomaly_score'], reverse=True)
        for player in sorted_suspicious[:10]:
            status = "VAC BANNED" if player['vac_banned'] else "SUSPICIOUS"
            print(f"  - {player['username']}: {player['anomaly_score']:.2f} [{status}]")
    
    print(f"\n🤖 Ready for ML training with {len(dataset)} players!")
    return dataset

if __name__ == "__main__":
    collect_training_data()