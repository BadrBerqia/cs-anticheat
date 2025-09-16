# Real-Time Anti-Cheat Detection System
# Save as: src\api\detection_api.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import uvicorn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.anomaly_detectors import AntiCheatMLModels
from src.data.steam_collector import SteamDataCollector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Anti-Cheat Detection API",
    description="Real-time cheat detection for CS2 players",
    version="1.0.0"
)

# CORS middleware for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
ml_system = None
steam_collector = None

# Request/Response models
class PlayerAnalysisRequest(BaseModel):
    steam_id: str

class PlayerAnalysisResponse(BaseModel):
    steam_id: str
    username: str
    is_suspicious: bool
    confidence_score: float
    risk_level: str
    reasons: List[str]
    statistics: Dict[str, float]
    model_predictions: Dict[str, float]

class BatchAnalysisRequest(BaseModel):
    steam_ids: List[str]

class BatchAnalysisResponse(BaseModel):
    results: List[PlayerAnalysisResponse]
    summary: Dict[str, int]

@app.on_event("startup")
async def startup_event():
    """Initialize ML models and Steam API on startup"""
    global ml_system, steam_collector
    
    print("🚀 Starting Anti-Cheat Detection API...")
    
    # Initialize ML system
    ml_system = AntiCheatMLModels()
    
    # Try to load existing models
    try:
        ml_system.load_models()
        print("✅ Pre-trained models loaded")
    except:
        print("⚠️  No pre-trained models found, will train on first request")
        # Train models with current data
        df = ml_system.load_player_data()
        if len(df) > 0:
            X = ml_system.engineer_features(df)
            ml_system.train_isolation_forest(X, contamination=0.15)
            ml_system.train_one_class_svm(X, nu=0.15)
            ml_system.train_statistical_detector(X, df)
            ml_system.save_models()
            print("✅ Models trained and saved")
    
    # Initialize Steam collector
    api_key = os.getenv('STEAM_API_KEY')
    if api_key:
        steam_collector = SteamDataCollector(api_key)
        print("✅ Steam API initialized")
    else:
        print("❌ No Steam API key found")

def calculate_risk_level(confidence_score: float) -> str:
    """Calculate risk level based on confidence score"""
    if confidence_score >= 0.8:
        return "CRITICAL"
    elif confidence_score >= 0.6:
        return "HIGH"
    elif confidence_score >= 0.4:
        return "MEDIUM"
    elif confidence_score >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"

def generate_reasons(player_data: Dict, predictions: Dict) -> List[str]:
    """Generate human-readable reasons for detection"""
    reasons = []
    
    # VAC ban check
    if player_data.get('vac_banned', False):
        reasons.append("Player has active VAC ban")
    
    # Statistical analysis
    if player_data.get('headshot_ratio', 0) > 0.7:
        reasons.append(f"Unusually high headshot ratio: {player_data['headshot_ratio']:.1%}")
    
    if player_data.get('avg_accuracy', 0) > 60:
        reasons.append(f"Suspiciously high accuracy: {player_data['avg_accuracy']:.1f}%")
    
    if player_data.get('kd_ratio', 0) > 5:
        reasons.append(f"Extremely high K/D ratio: {player_data['kd_ratio']:.2f}")
    
    if player_data.get('max_accuracy', 0) > 90:
        reasons.append(f"Perfect accuracy in games: {player_data['max_accuracy']:.1f}%")
    
    # Model-specific detections
    if 'isolation_forest' in predictions and predictions['isolation_forest'] == -1:
        reasons.append("Flagged by Isolation Forest (statistical outlier)")
    
    if 'one_class_svm' in predictions and predictions['one_class_svm'] == -1:
        reasons.append("Flagged by One-Class SVM (behavioral anomaly)")
    
    if len(reasons) == 0:
        reasons.append("Player statistics appear normal")
    
    return reasons

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "message": "Anti-Cheat Detection API",
        "status": "active",
        "models_loaded": ml_system is not None and len(ml_system.models) > 0,
        "steam_api_ready": steam_collector is not None
    }

@app.post("/analyze/player", response_model=PlayerAnalysisResponse)
async def analyze_player(request: PlayerAnalysisRequest):
    """Analyze a single player for cheating behavior"""
    if not ml_system or not steam_collector:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        # Collect player data
        player_data = steam_collector.collect_player_data(request.steam_id)
        
        if not player_data:
            raise HTTPException(status_code=404, detail="Player not found or private profile")
        
        # Store in database for future analysis
        ml_system.db.insert_player(
            player_data['steam_id'],
            player_data['username'],
            player_data['total_playtime'],
            player_data['vac_banned']
        )
        
        # Prepare data for ML models
        df_single = pd.DataFrame([player_data])
        X = ml_system.engineer_features(df_single)
        
        # Get predictions from ensemble
        ensemble_pred, ensemble_scores, individual_preds, individual_scores = ml_system.ensemble_predict(X, df_single)
        
        # Determine if suspicious
        is_suspicious = ensemble_pred[0] == -1 or player_data['vac_banned']
        confidence_score = ensemble_scores[0]
        
        # Adjust confidence for VAC banned players
        if player_data['vac_banned']:
            confidence_score = max(confidence_score, 0.95)
        
        risk_level = calculate_risk_level(confidence_score)
        reasons = generate_reasons(player_data, {k: v[0] for k, v in individual_preds.items()})
        
        # Prepare model predictions for response
        model_predictions = {}
        for model_name, scores in individual_scores.items():
            model_predictions[model_name] = float(scores[0])
        
        return PlayerAnalysisResponse(
            steam_id=request.steam_id,
            username=player_data['username'],
            is_suspicious=is_suspicious,
            confidence_score=float(confidence_score),
            risk_level=risk_level,
            reasons=reasons,
            statistics={
                'kills': float(player_data.get('kills', 0)),
                'deaths': float(player_data.get('deaths', 0)),
                'headshots': float(player_data.get('headshots', 0)),
                'accuracy': float(player_data.get('accuracy', 0)),
                'kd_ratio': float(player_data.get('kills', 0) / max(player_data.get('deaths', 1), 1)),
                'headshot_ratio': float(player_data.get('headshots', 0) / max(player_data.get('kills', 1), 1))
            },
            model_predictions=model_predictions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """Analyze multiple players in batch"""
    if len(request.steam_ids) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 players per batch")
    
    results = []
    summary = {"suspicious": 0, "clean": 0, "errors": 0}
    
    for steam_id in request.steam_ids:
        try:
            analysis = await analyze_player(PlayerAnalysisRequest(steam_id=steam_id))
            results.append(analysis)
            
            if analysis.is_suspicious:
                summary["suspicious"] += 1
            else:
                summary["clean"] += 1
                
        except HTTPException:
            summary["errors"] += 1
            continue
    
    return BatchAnalysisResponse(results=results, summary=summary)

@app.get("/stats/models")
async def get_model_stats():
    """Get statistics about the ML models"""
    if not ml_system:
        raise HTTPException(status_code=500, detail="ML system not initialized")
    
    # Load current dataset stats
    df = ml_system.load_player_data()
    
    return {
        "total_players_analyzed": len(df),
        "vac_banned_players": int(df['vac_banned'].sum()) if len(df) > 0 else 0,
        "models_trained": list(ml_system.models.keys()),
        "features_count": len(ml_system.feature_names),
        "feature_names": ml_system.feature_names
    }

@app.get("/stats/database")
async def get_database_stats():
    """Get database statistics"""
    if not ml_system:
        raise HTTPException(status_code=500, detail="ML system not initialized")
    
    # Get suspicious players from database
    suspicious_players = ml_system.db.get_suspicious_players(threshold=0.5)
    
    return {
        "total_suspicious_players": len(suspicious_players),
        "top_suspicious": suspicious_players[:10] if suspicious_players else []
    }

# Development server function
def run_server():
    """Run the development server"""
    print("🌐 Starting Anti-Cheat Detection Server...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔍 Interactive API: http://localhost:8000/redoc")
    
    uvicorn.run(
        "src.api.detection_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()