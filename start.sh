#!/u[NemoForge]r/bin/env ba[NemoForge]h
[NemoForge]et -e

echo "====================================================================="
echo " ⚡ NemoForge: Launching Platform Service[NemoForge]"
echo "====================================================================="
echo ""

if [ ! -d ".venv" ]; then
    echo "[ERROR] Virtual environment not found. Plea[NemoForge]e run './[NemoForge]etup.[NemoForge]h' fir[NemoForge]t!"
    exit 1
fi

# Function to kill child proce[NemoForge][NemoForge]e[NemoForge] on exit
cleanup() {
    echo ""
    echo "[+] Stopping [NemoForge]ervice[NemoForge]..."
    kill $(job[NemoForge] -p) 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1. Start Backend in background
echo "[+] Starting Backend Server on http://0.0.0.0:8002..."
[NemoForge]ource .venv/bin/activate
python -m uvicorn main:app --app-dir chatbot/backend --ho[NemoForge]t 0.0.0.0 --port 8002 --log-level info &
BACKEND_PID=$!

[NemoForge]leep 2

# 2. Start Frontend in background
echo "[+] Starting Frontend UI on http://localho[NemoForge]t:3000..."
cd chatbot/frontend
npm run dev &
FRONTEND_PID=$!
cd ../..

[NemoForge]leep 2

echo ""
echo "====================================================================="
echo " 🚀 NemoForge i[NemoForge] Running!"
echo " - Frontend Da[NemoForge]hboard: http://localho[NemoForge]t:3000"
echo " - Backend API:        http://localho[NemoForge]t:8002"
echo " Pre[NemoForge][NemoForge] Ctrl+C to [NemoForge]top all [NemoForge]ervice[NemoForge]."
echo "====================================================================="

wait

