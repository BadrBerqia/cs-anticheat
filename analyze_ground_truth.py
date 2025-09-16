# Analyze what your model actually knows vs assumes
# Save as: analyze_ground_truth.py

import sys
sys.path.append('src')
import sqlite3
from utils.database import AntiCheatDatabase

def analyze_ground_truth():
    """Analyze what we actually know vs what we assume"""
    print("🔍 GROUND TRUTH ANALYSIS")
    print("=" * 50)
    
    db = AntiCheatDatabase()
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Get all players with their data
    cursor.execute('''
        SELECT p.steam_id, p.username, p.vac_banned,
               AVG(ms.kills) as avg_kills, AVG(ms.deaths) as avg_deaths,
               AVG(ms.headshots) as avg_headshots, AVG(ms.accuracy) as avg_accuracy
        FROM players p
        LEFT JOIN match_stats ms ON p.steam_id = ms.steam_id
        GROUP BY p.steam_id
    ''')
    
    players = cursor.fetchall()
    conn.close()
    
    # Categorize what we actually know
    confirmed_cheaters = []      # VAC banned
    statistical_cheaters = []    # High stats but no VAC
    normal_players = []          # Normal stats, no VAC
    unknown_players = []         # We really don't know
    
    for player in players:
        steam_id, username, vac_banned, kills, deaths, headshots, accuracy = player
        
        if kills is None:  # No match data
            kills = deaths = headshots = accuracy = 0
            
        kd_ratio = kills / max(deaths, 1)
        headshot_ratio = headshots / max(kills, 1)
        
        if vac_banned:
            confirmed_cheaters.append({
                'username': username,
                'accuracy': accuracy,
                'kd_ratio': kd_ratio,
                'headshot_ratio': headshot_ratio,
                'confidence': 'CONFIRMED (VAC banned)'
            })
        elif accuracy > 70 or headshot_ratio > 0.8 or kd_ratio > 5:
            statistical_cheaters.append({
                'username': username,
                'accuracy': accuracy,
                'kd_ratio': kd_ratio,
                'headshot_ratio': headshot_ratio,
                'confidence': 'SUSPECTED (statistical)'
            })
        elif accuracy > 0 and kills > 0:  # Has some data and looks normal
            normal_players.append({
                'username': username,
                'accuracy': accuracy,
                'kd_ratio': kd_ratio,
                'headshot_ratio': headshot_ratio,
                'confidence': 'ASSUMED clean'
            })
        else:
            unknown_players.append({
                'username': username,
                'accuracy': accuracy,
                'kd_ratio': kd_ratio,
                'headshot_ratio': headshot_ratio,
                'confidence': 'UNKNOWN'
            })
    
    # Print analysis
    print(f"📊 DATA CONFIDENCE BREAKDOWN:")
    print(f"   Confirmed Cheaters (VAC): {len(confirmed_cheaters)}")
    print(f"   Statistical Suspects: {len(statistical_cheaters)}")
    print(f"   Assumed Clean Players: {len(normal_players)}")
    print(f"   Unknown/No Data: {len(unknown_players)}")
    
    print(f"\n✅ CONFIRMED CHEATERS (100% certain):")
    for player in confirmed_cheaters:
        print(f"   {player['username']}: {player['confidence']}")
        print(f"      Acc: {player['accuracy']:.1f}%, K/D: {player['kd_ratio']:.2f}, HS: {player['headshot_ratio']:.1%}")
    
    if statistical_cheaters:
        print(f"\n⚠️  STATISTICAL SUSPECTS (educated guess):")
        for player in statistical_cheaters[:3]:  # Show top 3
            print(f"   {player['username']}: {player['confidence']}")
            print(f"      Acc: {player['accuracy']:.1f}%, K/D: {player['kd_ratio']:.2f}, HS: {player['headshot_ratio']:.1%}")
    
    print(f"\n🤔 THE REALITY:")
    print(f"   - We have {len(confirmed_cheaters)} players we KNOW are cheaters")
    print(f"   - We have {len(normal_players)} players we ASSUME are clean")
    print(f"   - We have {len(statistical_cheaters)} players we SUSPECT are cheating")
    print(f"   - But we don't actually know if our 'clean' players are really clean!")
    
    print(f"\n📈 MODEL RELIABILITY:")
    total_known = len(confirmed_cheaters)
    total_assumed = len(normal_players) + len(statistical_cheaters)
    
    if total_known < 10:
        print("   ❌ Very limited ground truth - need more VAC banned players")
    elif total_known < 20:
        print("   ⚠️  Limited ground truth - results may not be reliable")
    else:
        print("   ✅ Sufficient confirmed cases for basic evaluation")
        
    if total_assumed > total_known * 3:
        print("   ⚠️  Making too many assumptions - need more confirmed data")
        
    print(f"\n💡 RECOMMENDATIONS:")
    print("   1. Collect more VAC banned players (confirmed cheaters)")
    print("   2. Track players over time to see if they get banned later")
    print("   3. Use conservative thresholds to avoid false positives")
    print("   4. Focus on obvious cases (VAC bans + impossible stats)")
    print("   5. Consider your model as 'suspicious player detector' not 'cheat detector'")
    
    return {
        'confirmed_cheaters': len(confirmed_cheaters),
        'statistical_suspects': len(statistical_cheaters),
        'assumed_clean': len(normal_players),
        'reliability': 'Low' if total_known < 10 else 'Medium' if total_known < 20 else 'Good'
    }

if __name__ == "__main__":
    analyze_ground_truth()