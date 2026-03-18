@echo off
cd /d %~dp0

echo ============================================
echo   Kellanova AI - Starting Services
echo ============================================

echo.
echo [1/3] Starting Ollama...
start "KellanovaAI-Ollama" cmd /k "ollama serve"
timeout /t 2 /nobreak >nul

echo [2/3] Starting FastAPI Backend...
start "KellanovaAI-FastAPI" cmd /k "uvicorn api.main:app --reload --port 8000"
timeout /t 2 /nobreak >nul

echo [3/3] Starting Streamlit Dashboard...
start "KellanovaAI-Streamlit" cmd /k "streamlit run dashboard/app.py"

echo.
echo ============================================
echo   All services started!
echo.
echo   Dashboard : http://localhost:8501
echo   API       : http://localhost:8000
echo   API Docs  : http://localhost:8000/docs
echo ============================================
echo.
pause

