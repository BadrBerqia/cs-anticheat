# Anti-Cheat System Startup Script
Write-Host "🛡️ Starting Anti-Cheat Development Environment..." -ForegroundColor Green

# Navigate to project
Set-Location "C:\Users\Nasseem\cs-anticheat"

# Activate virtual environment
& "venv\Scripts\Activate.ps1"

# Show status
Write-Host "✅ Environment activated!" -ForegroundColor Green
Write-Host "📁 Project directory: C:\Users\Nasseem\cs-anticheat" -ForegroundColor Yellow
Write-Host ""
Write-Host "🚀 Quick commands:" -ForegroundColor Cyan
Write-Host "   python run_anticheat.py          # Run main system"
Write-Host "   python improve_model.py --collect-only --players 100"
Write-Host "   python evaluate_model.py         # Test accuracy"
