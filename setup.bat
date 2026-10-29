@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo  ⚡ NemoForge: Automated System Setup
echo =====================================================================
echo.

:: -----------------------------------------------------------------------
:: 1. Verify / Install Python (3.10+)
:: -----------------------------------------------------------------------
echo [*] Checking for Python...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto :MISSING_PYTHON

python -c "import sys; sys.exit(0 if sys.version_info[0] == 3 and sys.version_info[1] >= 10 else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto :OLD_PYTHON

for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo [+] %%v verified.
goto :CHECK_NODE

:MISSING_PYTHON
echo.
echo [!] Python was NOT found on your system.
where winget >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [*] Found Windows Package Manager (winget).
    echo [*] Attempting automated installation of Python 3.12...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    echo.
    echo [IMPORTANT] Python has been installed!
    echo Please CLOSE this terminal and reopen it (or run 'setup.bat' again) to refresh your system PATH.
    pause
    exit /b 1
)
echo [ERROR] Python 3.10+ is required to run NemoForge.
echo.
echo Please download and install Python from:
echo   https://www.python.org/downloads/
echo.
echo [IMPORTANT] When installing, make sure to check:
echo   [X] "Add python.exe to PATH"
echo.
start https://www.python.org/downloads/
pause
exit /b 1

:OLD_PYTHON
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set CURRENT_PY_VER=%%v
echo [ERROR] %CURRENT_PY_VER% is too old. Python 3.10+ is required.
echo Please install a modern version from https://www.python.org/downloads/
start https://www.python.org/downloads/
pause
exit /b 1

:: -----------------------------------------------------------------------
:: 2. Verify / Install Node.js (18+)
:: -----------------------------------------------------------------------
:CHECK_NODE
echo.
echo [*] Checking for Node.js...
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto :MISSING_NODE

node -e "process.exit(Number(process.versions.node.split('.')[0]) >= 18 ? 0 : 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto :OLD_NODE

for /f "tokens=*" %%v in ('node -v 2^>^&1') do echo [+] Node.js %%v verified.
goto :SETUP_VENV

:MISSING_NODE
echo.
echo [!] Node.js was NOT found on your system.
where winget >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [*] Found Windows Package Manager (winget).
    echo [*] Attempting automated installation of Node.js LTS...
    winget install -e --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
    echo.
    echo [IMPORTANT] Node.js has been installed!
    echo Please CLOSE this terminal and reopen it (or run 'setup.bat' again) to refresh your system PATH.
    pause
    exit /b 1
)
echo [ERROR] Node.js 18+ is required to run the NemoForge frontend.
echo.
echo Please download and install Node.js LTS from:
echo   https://nodejs.org/
echo.
start https://nodejs.org/
pause
exit /b 1

:OLD_NODE
for /f "tokens=*" %%v in ('node -v 2^>^&1') do set CURRENT_NODE_VER=%%v
echo [ERROR] Node.js %CURRENT_NODE_VER% is too old. Node.js 18+ is required.
echo Please install Node.js LTS from https://nodejs.org/
start https://nodejs.org/
pause
exit /b 1

:: -----------------------------------------------------------------------
:: 3. Setup Isolated Python Virtual Environment
:: -----------------------------------------------------------------------
:SETUP_VENV
echo.
if not exist ".venv" (
    echo [+] Creating Python virtual environment in .venv...
    python -m venv .venv
) else (
    echo [+] Virtual environment .venv already exists.
)

:: Link source directory into site-packages via .pth file
if exist ".venv\Lib\site-packages" (
    echo %~dp0src> .venv\Lib\site-packages\nemoforge.pth
)

:: -----------------------------------------------------------------------
:: 4. Install Python Dependencies
:: -----------------------------------------------------------------------
echo [+] Installing backend dependencies in .venv...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r chatbot\backend\requirements.txt
echo [+] Backend Python dependencies installed successfully.

:: -----------------------------------------------------------------------
:: 5. Install Frontend Dependencies
:: -----------------------------------------------------------------------
echo [+] Installing frontend (React + Vite) dependencies...
cd chatbot\frontend
call npm install --silent
cd ..\..
echo [+] Frontend dependencies installed successfully.

:: -----------------------------------------------------------------------
:: 6. Environment Configuration
:: -----------------------------------------------------------------------
if not exist ".env.local" (
    echo [+] Generating .env.local from .env.example...
    copy .env.example .env.local >nul
    echo.
    echo -----------------------------------------------------------------
    echo [ACTION] You can optionally add your Nebius API Key in .env.local:
    echo    NEBIUS_API_KEY=your_key_here
    echo (Or simply enter it in the web UI under Settings!)
    echo -----------------------------------------------------------------
) else (
    echo [+] .env.local already exists.
)

echo.
echo =====================================================================
echo  🎉 NemoForge Setup Complete!
echo  Run 'start.bat' (or double-click it) to launch NemoForge.
echo =====================================================================
