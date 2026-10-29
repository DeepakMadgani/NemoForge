@echo off
setlocal

echo =====================================================================
echo  [NemoForge] Launching Platform Services
echo =====================================================================
echo.

if not exist ".venv" (
    echo [ERROR] Virtual environment not found. Please run 'setup.bat' first!
    exit /b 1
)

:: 1. Start Backend in background window
echo [+] Starting Backend Server on http://0.0.0.0:8002...
start "NemoForge Backend (Port 8002)" cmd /k ".venv\Scripts\activate.bat && python -m uvicorn main:app --app-dir chatbot\backend --host 0.0.0.0 --port 8002 --log-level info"

:: 2. Wait 2 seconds
timeout /t 2 /nobreak >nul

:: 3. Start Frontend in background window
echo [+] Starting Frontend UI on http://localhost:3000...
start "NemoForge Frontend (Port 3000)" cmd /k "cd chatbot\frontend && npm run dev"

:: 4. Wait 3 seconds and open browser
timeout /t 3 /nobreak >nul
echo [+] Opening NemoForge Dashboard in browser...
start http://localhost:3000

echo.
echo =====================================================================
echo  [NemoForge] Running!
echo  - Frontend Dashboard: http://localhost:3000
echo  - Backend API:        http://localhost:8002
echo =====================================================================
