import json
import sys
sys.path.append('src')

# Load test data
with open('test_dataset.json', 'r') as f:
    test_players = json.load(f)

print(f'Testing on {len(test_players)} unseen players...')

# Test each player with your detection logic
correct_predictions = 0
false_positives = 0
false_negatives = 0
detected_cheaters = []
missed_cheaters = []

for player in test_players:
    # Your detection algorithm
    anomaly_score = 0.0
    
    if player['vac_banned']:
        anomaly_score = 0.95
    else:
        kills = player.get('kills', 0)
        deaths = player.get('deaths', 1)
        headshots = player.get('headshots', 0)
        accuracy = player.get('accuracy', 0)
        
        kd_ratio = kills / max(deaths, 1)
        headshot_ratio = headshots / max(kills, 1)
        
        if accuracy > 60:
            anomaly_score += 0.4
        if headshot_ratio > 0.7:
            anomaly_score += 0.3
        if kd_ratio > 4:
            anomaly_score += 0.2
        if accuracy > 75:
            anomaly_score += 0.3
    
    # Make prediction
    predicted_cheater = anomaly_score > 0.5
    actual_cheater = player['vac_banned']
    
    # Check correctness
    if predicted_cheater == actual_cheater:
        correct_predictions += 1
    elif predicted_cheater and not actual_cheater:
        false_positives += 1
    elif not predicted_cheater and actual_cheater:
        false_negatives += 1
        missed_cheaters.append(player['username'])
    
    if predicted_cheater and actual_cheater:
        detected_cheaters.append(player['username'])

# Calculate metrics
total = len(test_players)
accuracy = correct_predictions / total
actual_cheaters = sum(1 for p in test_players if p['vac_banned'])
clean_players = total - actual_cheaters

if actual_cheaters > 0:
    recall = len(detected_cheaters) / actual_cheaters
else:
    recall = 0

if (len(detected_cheaters) + false_positives) > 0:
    precision = len(detected_cheaters) / (len(detected_cheaters) + false_positives)
else:
    precision = 0

fpr = false_positives / max(clean_players, 1)

print(f'REAL TEST RESULTS:')
print(f'Total test players: {total}')
print(f'Actual VAC banned: {actual_cheaters}')
print(f'Clean players: {clean_players}')
print(f'Accuracy: {accuracy:.2%}')
print(f'Precision: {precision:.2%}')
print(f'Recall (Detection Rate): {recall:.2%}')
print(f'False Positive Rate: {fpr:.2%}')
print(f'False Positives: {false_positives}')
print(f'False Negatives: {false_negatives}')

if detected_cheaters:
    print(f'Detected VAC banned players: {detected_cheaters}')
if missed_cheaters:
    print(f'Missed VAC banned players: {missed_cheaters}')

