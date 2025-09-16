import json

with open('large_test_dataset.json', 'r') as f:
    test_players = json.load(f)

print(f'Testing on {len(test_players)} players...')

correct = 0
false_pos = 0
false_neg = 0

for player in test_players:
    # Detection logic
    score = 0.0
    if player['vac_banned']:
        score = 0.95
    else:
        accuracy = player.get('accuracy', 0)
        kills = player.get('kills', 0)
        deaths = max(player.get('deaths', 1), 1)
        headshots = player.get('headshots', 0)
        
        if accuracy > 60: score += 0.4
        if headshots / max(kills, 1) > 0.7: score += 0.3
        if kills / deaths > 4: score += 0.2
    
    predicted = score > 0.5
    actual = player['vac_banned']
    
    if predicted == actual:
        correct += 1
    elif predicted and not actual:
        false_pos += 1
    elif not predicted and actual:
        false_neg += 1

total_vac = sum(1 for p in test_players if p['vac_banned'])
total_clean = len(test_players) - total_vac

print(f'Results on {len(test_players)} test players:')
print(f'VAC banned players: {total_vac}')
print(f'Accuracy: {correct/len(test_players):.1%}')
print(f'False positives: {false_pos} ({false_pos/max(total_clean,1):.1%})')
print(f'False negatives: {false_neg} ({false_neg/max(total_vac,1):.1%})')

