# Ticket-to-Code Chatbot Backend - Setup Script
# Automates installation and initial configuration

Write-Host "🚀 Setting up Ticket-to-Code Chatbot Backend..." -ForegroundColor Cyan

# Navigate to backend directory
$BackendPath = Join-Path $PSScriptRoot "chatbot\backend"
if (!(Test-Path $BackendPath)) {
    New-Item -ItemType Directory -Path $BackendPath -Force | Out-Null
}
Set-Location $BackendPath

# Check Python installation
Write-Host "`n📦 Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found. Please install Python 3.10+`n" -ForegroundColor Red
    exit 1
}
Write-Host "✅ $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host "`n🔧 Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "⚠️  Virtual environment already exists, skipping..." -ForegroundColor Yellow
} else {
    python -m venv .venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n✨ Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "`n📥 Installing Python dependencies..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt --quiet
    Write-Host "✅ Core dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠️  requirements.txt not found" -ForegroundColor Yellow
}

if (Test-Path "requirements-playwright.txt") {
    pip install -r requirements-playwright.txt --quiet
    Write-Host "✅ Playwright dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠️  requirements-playwright.txt not found" -ForegroundColor Yellow
}

# Install Playwright browsers
Write-Host "`n🎭 Installing Playwright browsers..." -ForegroundColor Yellow
playwright install chromium
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Playwright chromium browser installed" -ForegroundColor Green
} else {
    Write-Host "⚠️  Playwright installation had issues" -ForegroundColor Yellow
}

# Create .env file if it doesn't exist
Write-Host "`n⚙️  Setting up configuration..." -ForegroundColor Yellow
if (!(Test-Path ".env")) {
    $envContent = @"
# ValueEdge Configuration
VALUEEDGE_URL=https://valueedge.your-company.com
VALUEEDGE_USERNAME=your-username
VALUEEDGE_PASSWORD=your-password

# Server Configuration
HOST=0.0.0.0
PORT=8000
"@
    $envContent | Out-File -FilePath ".env" -Encoding utf8
    Write-Host "✅ Created .env file (please update with your credentials)" -ForegroundColor Green
} else {
    Write-Host "⚠️  .env file already exists" -ForegroundColor Yellow
}

# Create workspace directories
Write-Host "`n📁 Creating workspace directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "workspace" -Force | Out-Null
New-Item -ItemType Directory -Path "workspace\kb" -Force | Out-Null
Write-Host "✅ Workspace directories created" -ForegroundColor Green

# Summary
Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Cyan

Write-Host "`n📝 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Update .env file with your ValueEdge credentials" -ForegroundColor White
Write-Host "2. Run the server:" -ForegroundColor White
Write-Host "   python main.py" -ForegroundColor Cyan
Write-Host "`n3. Test Playwright scraper:" -ForegroundColor White
Write-Host "   python -c `"import asyncio; from services.valueedge_scraper import fetch_valueedge_ticket_example; asyncio.run(fetch_valueedge_ticket_example())`"" -ForegroundColor Cyan
Write-Host "`n4. Access API docs at:" -ForegroundColor White
Write-Host "   http://localhost:8000/docs" -ForegroundColor Cyan

Write-Host "`n🎉 Happy coding!" -ForegroundColor Green
