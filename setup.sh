#!/usr/bin/env bash
set -e

echo "====================================================================="
echo " ⚡ NemoForge: Automated System Setup"
echo "====================================================================="
echo ""

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3.10+ is required but not found."
    exit 1
fi
echo "[+] Python found: $(python3 --version)"

# 2. Check Node.js
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js 18+ is required but not found."
    exit 1
fi
echo "[+] Node.js found: $(node --version)"

# 3. Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[+] Creating Python virtual environment in .venv..."
    python3 -m venv .venv
else
    echo "[+] Virtual environment .venv already exists."
fi

# 4. Install Python Dependencies
echo "[+] Installing Python dependencies..."
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -e . --quiet || pip install -r chatbot/backend/requirements.txt --quiet
echo "[+] Python dependencies installed successfully."

# 5. Install Frontend Dependencies
echo "[+] Installing Frontend (React + Vite) dependencies..."
cd chatbot/frontend
npm install --silent
cd ../..
echo "[+] Frontend dependencies installed successfully."

# 6. Environment Configuration
if [ ! -f ".env.local" ]; then
    echo "[+] Creating .env.local from .env.example..."
    cp .env.example .env.local
    echo ""
    echo "-----------------------------------------------------------------"
    echo "[ACTION REQUIRED] Please open '.env.local' and add your Nebius API Key:"
    echo "   NEBIUS_API_KEY=your_key_here"
    echo "(Alternatively, you can enter it directly in the UI Settings!)"
    echo "-----------------------------------------------------------------"
else
    echo "[+] .env.local already configured."
fi

echo ""
echo "====================================================================="
echo " 🎉 Setup Complete! Run './start.sh' to launch NemoForge."
echo "====================================================================="

