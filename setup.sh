#!/usr/bin/env bash
set -e

echo "====================================================================="
echo " ⚡ NemoForge: Automated System Setup"
echo "====================================================================="
echo ""

# -----------------------------------------------------------------------
# 1. Verify / Install Python (3.10+)
# -----------------------------------------------------------------------
echo "[*] Checking for Python..."
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "[!] Python 3 was NOT found on your system."
    echo "[ERROR] Python 3.10+ is required to run NemoForge."
    echo ""
    if command -v brew &> /dev/null; then
        echo "Detected Homebrew. You can install Python by running:"
        echo "   brew install python"
    elif command -v apt-get &> /dev/null; then
        echo "Detected Debian/Ubuntu. You can install Python by running:"
        echo "   sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    elif command -v dnf &> /dev/null; then
        echo "Detected Fedora/RHEL. You can install Python by running:"
        echo "   sudo dnf install -y python3 python3-pip"
    elif command -v pacman &> /dev/null; then
        echo "Detected Arch Linux. You can install Python by running:"
        echo "   sudo pacman -S python python-pip"
    else
        echo "Please install Python 3.10+ from: https://www.python.org/downloads/"
    fi
    exit 1
fi

# Check Python version >= 3.10
python3 -c "import sys; sys.exit(0 if sys.version_info[0] == 3 and sys.version_info[1] >= 10 else 1)" 2>/dev/null || {
    CURRENT_PY=$(python3 --version 2>&1)
    echo "[ERROR] $CURRENT_PY is too old. Python 3.10+ is required."
    echo "Please upgrade Python from: https://www.python.org/downloads/"
    exit 1
}
echo "[+] $(python3 --version) verified."

# -----------------------------------------------------------------------
# 2. Verify / Install Node.js (18+)
# -----------------------------------------------------------------------
echo "[*] Checking for Node.js..."
if ! command -v node &> /dev/null; then
    echo ""
    echo "[!] Node.js was NOT found on your system."
    echo "[ERROR] Node.js 18+ is required to run the NemoForge frontend."
    echo ""
    if command -v brew &> /dev/null; then
        echo "Detected Homebrew. You can install Node.js by running:"
        echo "   brew install node"
    elif command -v apt-get &> /dev/null; then
        echo "Detected Debian/Ubuntu. You can install Node.js LTS by running:"
        echo "   curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
        echo "   sudo apt install -y nodejs"
    else
        echo "Please install Node.js LTS from: https://nodejs.org/"
    fi
    exit 1
fi

# Check Node.js version >= 18
node -e "process.exit(Number(process.versions.node.split('.')[0]) >= 18 ? 0 : 1)" 2>/dev/null || {
    CURRENT_NODE=$(node -v 2>&1)
    echo "[ERROR] Node.js $CURRENT_NODE is too old. Node.js 18+ is required."
    echo "Please upgrade Node.js LTS from: https://nodejs.org/"
    exit 1
}
echo "[+] Node.js $(node -v) verified."

# -----------------------------------------------------------------------
# 3. Setup Isolated Python Virtual Environment
# -----------------------------------------------------------------------
echo ""
if [ ! -d ".venv" ]; then
    echo "[+] Creating Python virtual environment in .venv..."
    python3 -m venv .venv
else
    echo "[+] Virtual environment .venv already exists."
fi

# Link source directory into site-packages via .pth file
SITE_PACKAGES=$(find .venv/lib -name "site-packages" 2>/dev/null | head -n 1)
if [ -n "$SITE_PACKAGES" ]; then
    echo "$(pwd)/src" > "$SITE_PACKAGES/nemoforge.pth"
fi

# -----------------------------------------------------------------------
# 4. Install Python Dependencies
# -----------------------------------------------------------------------
echo "[+] Installing backend dependencies in .venv..."
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r chatbot/backend/requirements.txt
echo "[+] Backend Python dependencies installed successfully."

# -----------------------------------------------------------------------
# 5. Install Frontend Dependencies
# -----------------------------------------------------------------------
echo "[+] Installing frontend (React + Vite) dependencies..."
cd chatbot/frontend
npm install --silent
cd ../..
echo "[+] Frontend dependencies installed successfully."

# -----------------------------------------------------------------------
# 6. Environment Configuration
# -----------------------------------------------------------------------
if [ ! -f ".env.local" ]; then
    echo "[+] Generating .env.local from .env.example..."
    cp .env.example .env.local
    echo ""
    echo "-----------------------------------------------------------------"
    echo "[ACTION] You can optionally add your Nebius API Key in .env.local:"
    echo "   NEBIUS_API_KEY=your_key_here"
    echo "(Or simply enter it in the web UI under Settings!)"
    echo "-----------------------------------------------------------------"
else
    echo "[+] .env.local already exists."
fi

echo ""
echo "====================================================================="
echo " 🎉 NemoForge Setup Complete!"
echo " Run './start.sh' to launch NemoForge."
echo "====================================================================="
