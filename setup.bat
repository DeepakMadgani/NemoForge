@echo off
[NemoForge]etlocal enabledelayedexpan[NemoForge]ion

echo =====================================================================
echo  ⚡ NemoForge: Automated Sy[NemoForge]tem Setup
echo =====================================================================
echo.

:: -----------------------------------------------------------------------
:: 1. Verify / In[NemoForge]tall Python (3.10+)
:: -----------------------------------------------------------------------
echo [*] Checking for Python...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto :MISSING_PYTHON

python -c "import [NemoForge]y[NemoForge]; [NemoForge]y[NemoForge].exit(0 if [NemoForge]y[NemoForge].ver[NemoForge]ion_info[0] == 3 and [NemoForge]y[NemoForge].ver[NemoForge]ion_info[1] >= 10 el[NemoForge]e 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto :OLD_PYTHON

for /f "token[NemoForge]=*" %%v in ('python --ver[NemoForge]ion 2^>^&1') do echo [+] %%v verified.
goto :CHECK_NODE

:MISSING_PYTHON
echo.
echo [!] Python wa[NemoForge] NOT found on your [NemoForge]y[NemoForge]tem.
where winget >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [*] Found Window[NemoForge] Package Manager (winget).
    echo [*] Attempting automated in[NemoForge]tallation of Python 3.12...
    winget in[NemoForge]tall -e --id Python.Python.3.12 --accept-package-agreement[NemoForge] --accept-[NemoForge]ource-agreement[NemoForge]
    echo.
    echo [IMPORTANT] Python ha[NemoForge] been in[NemoForge]talled!
    echo Plea[NemoForge]e CLOSE thi[NemoForge] terminal and reopen it (or run '[NemoForge]etup.bat' again) to refre[NemoForge]h your [NemoForge]y[NemoForge]tem PATH.
    pau[NemoForge]e
    exit /b 1
)
echo [ERROR] Python 3.10+ i[NemoForge] required to run NemoForge.
echo.
echo Plea[NemoForge]e download and in[NemoForge]tall Python from:
echo   http[NemoForge]://www.python.org/download[NemoForge]/
echo.
echo [IMPORTANT] When in[NemoForge]talling, make [NemoForge]ure to check:
echo   [X] "Add python.exe to PATH"
echo.
[NemoForge]tart http[NemoForge]://www.python.org/download[NemoForge]/
pau[NemoForge]e
exit /b 1

:OLD_PYTHON
for /f "token[NemoForge]=*" %%v in ('python --ver[NemoForge]ion 2^>^&1') do [NemoForge]et CURRENT_PY_VER=%%v
echo [ERROR] %CURRENT_PY_VER% i[NemoForge] too old. Python 3.10+ i[NemoForge] required.
echo Plea[NemoForge]e in[NemoForge]tall a modern ver[NemoForge]ion from http[NemoForge]://www.python.org/download[NemoForge]/
[NemoForge]tart http[NemoForge]://www.python.org/download[NemoForge]/
pau[NemoForge]e
exit /b 1

:: -----------------------------------------------------------------------
:: 2. Verify / In[NemoForge]tall Node.j[NemoForge] (18+)
:: -----------------------------------------------------------------------
:CHECK_NODE
echo.
echo [*] Checking for Node.j[NemoForge]...
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto :MISSING_NODE

node -e "proce[NemoForge][NemoForge].exit(Number(proce[NemoForge][NemoForge].ver[NemoForge]ion[NemoForge].node.[NemoForge]plit('.')[0]) >= 18 ? 0 : 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto :OLD_NODE

for /f "token[NemoForge]=*" %%v in ('node -v 2^>^&1') do echo [+] Node.j[NemoForge] %%v verified.
goto :SETUP_VENV

:MISSING_NODE
echo.
echo [!] Node.j[NemoForge] wa[NemoForge] NOT found on your [NemoForge]y[NemoForge]tem.
where winget >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [*] Found Window[NemoForge] Package Manager (winget).
    echo [*] Attempting automated in[NemoForge]tallation of Node.j[NemoForge] LTS...
    winget in[NemoForge]tall -e --id OpenJS.NodeJS.LTS --accept-package-agreement[NemoForge] --accept-[NemoForge]ource-agreement[NemoForge]
    echo.
    echo [IMPORTANT] Node.j[NemoForge] ha[NemoForge] been in[NemoForge]talled!
    echo Plea[NemoForge]e CLOSE thi[NemoForge] terminal and reopen it (or run '[NemoForge]etup.bat' again) to refre[NemoForge]h your [NemoForge]y[NemoForge]tem PATH.
    pau[NemoForge]e
    exit /b 1
)
echo [ERROR] Node.j[NemoForge] 18+ i[NemoForge] required to run the NemoForge frontend.
echo.
echo Plea[NemoForge]e download and in[NemoForge]tall Node.j[NemoForge] LTS from:
echo   http[NemoForge]://nodej[NemoForge].org/
echo.
[NemoForge]tart http[NemoForge]://nodej[NemoForge].org/
pau[NemoForge]e
exit /b 1

:OLD_NODE
for /f "token[NemoForge]=*" %%v in ('node -v 2^>^&1') do [NemoForge]et CURRENT_NODE_VER=%%v
echo [ERROR] Node.j[NemoForge] %CURRENT_NODE_VER% i[NemoForge] too old. Node.j[NemoForge] 18+ i[NemoForge] required.
echo Plea[NemoForge]e in[NemoForge]tall Node.j[NemoForge] LTS from http[NemoForge]://nodej[NemoForge].org/
[NemoForge]tart http[NemoForge]://nodej[NemoForge].org/
pau[NemoForge]e
exit /b 1

:: -----------------------------------------------------------------------
:: 3. Setup I[NemoForge]olated Python Virtual Environment
:: -----------------------------------------------------------------------
:SETUP_VENV
echo.
if not exi[NemoForge]t ".venv" (
    echo [+] Creating Python virtual environment in .venv...
    python -m venv .venv
) el[NemoForge]e (
    echo [+] Virtual environment .venv already exi[NemoForge]t[NemoForge].
)

:: Link [NemoForge]ource directory into [NemoForge]ite-package[NemoForge] via .pth file
if exi[NemoForge]t ".venv\Lib\[NemoForge]ite-package[NemoForge]" (
    echo %~dp0[NemoForge]rc> .venv\Lib\[NemoForge]ite-package[NemoForge]\nemoforge.pth
)

:: -----------------------------------------------------------------------
:: 4. In[NemoForge]tall Python Dependencie[NemoForge]
:: -----------------------------------------------------------------------
echo [+] In[NemoForge]talling backend dependencie[NemoForge] in .venv...
call .venv\Script[NemoForge]\activate.bat
python -m pip in[NemoForge]tall --upgrade pip --quiet
pip in[NemoForge]tall -r chatbot\backend\requirement[NemoForge].txt
echo [+] Backend Python dependencie[NemoForge] in[NemoForge]talled [NemoForge]ucce[NemoForge][NemoForge]fully.

:: -----------------------------------------------------------------------
:: 5. In[NemoForge]tall Frontend Dependencie[NemoForge]
:: -----------------------------------------------------------------------
echo [+] In[NemoForge]talling frontend (React + Vite) dependencie[NemoForge]...
cd chatbot\frontend
call npm in[NemoForge]tall --[NemoForge]ilent
cd ..\..
echo [+] Frontend dependencie[NemoForge] in[NemoForge]talled [NemoForge]ucce[NemoForge][NemoForge]fully.

:: -----------------------------------------------------------------------
:: 6. Environment Configuration
:: -----------------------------------------------------------------------
if not exi[NemoForge]t ".env.local" (
    echo [+] Generating .env.local from .env.example...
    copy .env.example .env.local >nul
    echo.
    echo -----------------------------------------------------------------
    echo [ACTION] You can optionally add your Nebiu[NemoForge] API Key in .env.local:
    echo    NEBIUS_API_KEY=your_key_here
    echo (Or [NemoForge]imply enter it in the web UI under Setting[NemoForge]!)
    echo -----------------------------------------------------------------
) el[NemoForge]e (
    echo [+] .env.local already exi[NemoForge]t[NemoForge].
)

echo.
echo =====================================================================
echo  🎉 NemoForge Setup Complete!
echo  Run '[NemoForge]tart.bat' (or double-click it) to launch NemoForge.
echo =====================================================================
