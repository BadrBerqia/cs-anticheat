# Save this as src\utils\database.py
import sqlite3
import os
import json
from datetime import datetime

class AntiCheatDatabase:
    def __init__(self, db_path="anticheat.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Players table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                steam_id TEXT PRIMARY KEY,
                username TEXT,
                total_playtime INTEGER DEFAULT 0,
                vac_banned BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Match stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS match_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                steam_id TEXT,
                kills INTEGER DEFAULT 0,
                deaths INTEGER DEFAULT 0,
                headshots INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0.0,
                anomaly_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (steam_id) REFERENCES players (steam_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")
    
    def insert_player(self, steam_id, username, playtime=0, vac_banned=False):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO players 
                (steam_id, username, total_playtime, vac_banned)
                VALUES (?, ?, ?, ?)
            ''', (steam_id, username, playtime, vac_banned))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error inserting player: {e}")
            return False

# Test function
if __name__ == "__main__":
    print("🧪 Testing database...")
    db = AntiCheatDatabase()
    
    # Test data
    if db.insert_player("76561198000000001", "TestPlayer", 1500, False):
        print("✅ Test player inserted successfully!")
    
    print("🎉 Database ready for Steam data!")