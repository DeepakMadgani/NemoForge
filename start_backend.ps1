$NEMOFORGE_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $NEMOFORGE_DIR

# Kill existing process on port 8002 if running
try {
    $conns = Get-NetTCPConnection -LocalPort 8002 -ErrorAction SilentlyContinue
    if ($conns) {
        foreach ($c in $conns) {
            Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
} catch {}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$NEMOFORGE_DIR\src;$NEMOFORGE_DIR"

# Universal JDK discovery
if (-not (Test-Path "$env:JAVA_HOME\bin\java.exe")) {
    $validJdk = Get-ChildItem "C:\Program Files\Microsoft\jdk*", "C:\Program Files\Java\jdk*", "C:\Program Files\Eclipse Adoptium\jdk*" -ErrorAction SilentlyContinue | Where-Object { Test-Path "$_\bin\java.exe" } | Sort-Object Name -Descending | Select-Object -First 1
    if ($validJdk) {
        $env:JAVA_HOME = $validJdk.FullName
        $env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
    }
}

Write-Host "🚀 Launching NemoForge Backend on port 8002..." -ForegroundColor Green
& "$NEMOFORGE_DIR\.venv\Scripts\python.exe" -m uvicorn main:app --app-dir "$NEMOFORGE_DIR\chatbot\backend" --host 0.0.0.0 --port 8002 --log-level info

