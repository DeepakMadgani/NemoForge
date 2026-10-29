#!/usr/bin/env bash
set -e

echo "====================================================================="
echo " ⚡ NemoForge: Launching Platform Services"
echo "====================================================================="
echo ""

if [ ! -d ".venv" ]; then
    echo "[ERROR] Virtual environment not found. Please run './setup.sh' first!"
    exit 1
fi

# Function to kill child processes on exit
cleanup() {
    echo ""
    echo "[+] Stopping services..."
    kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1. Start Backend in background
echo "[+] Starting Backend Server on http://0.0.0.0:8002..."
source .venv/bin/activate
python -m uvicorn main:app --app-dir chatbot/backend --host 0.0.0.0 --port 8002 --log-level info &
BACKEND_PID=$!

sleep 2

# 2. Start Frontend in background
echo "[+] Starting Frontend UI on http://localhost:3000..."
cd chatbot/frontend
npm run dev &
FRONTEND_PID=$!
cd ../..

sleep 2

echo ""
echo "====================================================================="
echo " 🚀 NemoForge is Running!"
echo " - Frontend Dashboard: http://localhost:3000"
echo " - Backend API:        http://localhost:8002"
echo " Press Ctrl+C to stop all services."
echo "====================================================================="

wait

