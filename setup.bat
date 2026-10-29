@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo  ⚡ NemoForge: Automated System Setup
echo =====================================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.10+ is required but not found in PATH.
    echo Please install Python from https://python.org and add it to PATH.
    exit /b 1
)
echo [+] Python found.

:: 2. Check Node.js
node --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js 18+ is required but not found in PATH.
    echo Please install Node.js from https://nodejs.org.
    exit /b 1
)
echo [+] Node.js found.

:: 3. Setup Virtual Environment
if not exist ".venv" (
    echo [+] Creating Python virtual environment in .venv...
    python -m venv .venv
) else (
    echo [+] Virtual environment .venv already exists.
)

:: 4. Install Python Dependencies
echo [+] Installing Python dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -e . --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] 'pip install -e .' had issues, installing from requirements.txt...
    pip install -r chatbot\backend\requirements.txt --quiet
)
echo [+] Python dependencies installed successfully.

:: 5. Install Frontend Dependencies
echo [+] Installing Frontend (React + Vite) dependencies...
cd chatbot\frontend
call npm install --silent
cd ..\..
echo [+] Frontend dependencies installed successfully.

:: 6. Environment Configuration
if not exist ".env.local" (
    echo [+] Creating .env.local from .env.example...
    copy .env.example .env.local >nul
    echo.
    echo -----------------------------------------------------------------
    echo [ACTION REQUIRED] Please open '.env.local' and add your Nebius API Key:
    echo    NEBIUS_API_KEY=your_key_here
    echo (Alternatively, you can enter it directly in the UI Settings!)
    echo -----------------------------------------------------------------
) else (
    echo [+] .env.local already configured.
)

echo.
echo =====================================================================
echo  🎉 Setup Complete! Run 'start.bat' to launch NemoForge.
echo =====================================================================

