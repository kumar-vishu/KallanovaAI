@echo off

echo ============================================
echo   Kellanova AI - Stopping Services
echo ============================================

echo.
echo [1/3] Stopping Streamlit...
taskkill /FI "WINDOWTITLE eq KellanovaAI-Streamlit*" /T /F >nul 2>&1

echo [2/3] Stopping FastAPI Backend...
taskkill /FI "WINDOWTITLE eq KellanovaAI-FastAPI*" /T /F >nul 2>&1

echo [3/3] Stopping Ollama...
taskkill /FI "WINDOWTITLE eq KellanovaAI-Ollama*" /T /F >nul 2>&1

echo.
echo ============================================
echo   All services stopped.
echo ============================================
echo.
pause

