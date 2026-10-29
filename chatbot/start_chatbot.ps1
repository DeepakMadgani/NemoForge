# ============================================================
# START TICKET-TO-CODE CHATBOT (Backend + Frontend)
# ============================================================

$REPO_ROOT = (Resolve-Path "$PSScriptRoot\..").Path
$BACKEND_DIR = "$REPO_ROOT\chatbot\backend"
$FRONTEND_DIR = "$REPO_ROOT\chatbot\frontend"
$SRC_DIR = "$REPO_ROOT\src;$REPO_ROOT"
$BACKEND_PORT = 8002
$env:PYTHONIOENCODING = "utf-8"

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host " TICKET-TO-CODE CHATBOT STARTUP" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host ""
Write-Host "Starting backend on port $BACKEND_PORT..." -ForegroundColor Yellow

# Free port if needed
Get-NetTCPConnection -LocalPort $BACKEND_PORT -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1

# Start backend in a new window
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "`$env:PYTHONIOENCODING='utf-8'; " +
    "`$env:PYTHONPATH='$SRC_DIR'; " +
    "Set-Location '$REPO_ROOT'; " +
    "python chatbot/backend/main.py"
)

Write-Host "  Backend starting in new window..." -ForegroundColor Green
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "Starting frontend on port 3000..." -ForegroundColor Yellow

# Start frontend in a new window
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$FRONTEND_DIR'; npm.cmd run dev"
)

Write-Host "  Frontend starting in new window..." -ForegroundColor Green
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host " CHATBOT READY" -ForegroundColor Cyan
Write-Host "  Frontend:  http://localhost:3000" -ForegroundColor White
Write-Host "  Backend:   http://localhost:$BACKEND_PORT" -ForegroundColor White
Write-Host "  Health:    http://localhost:$BACKEND_PORT/api/health" -ForegroundColor White
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "HOW TO GIVE A TICKET:" -ForegroundColor Yellow
Write-Host "  1. Select a project in the left panel" -ForegroundColor White
Write-Host "  2. Type your task in the chat, e.g.:" -ForegroundColor White
Write-Host "     'implement add area endpoint in AreaService.java'" -ForegroundColor Gray
Write-Host "     'fix null pointer in PaymentService'" -ForegroundColor Gray
Write-Host "     'add logging to database queries'" -ForegroundColor Gray
Write-Host "  3. Watch live execution in Transparent Workflow view" -ForegroundColor White
Write-Host ""
