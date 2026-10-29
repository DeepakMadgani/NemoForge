#!/u[NemoForge]r/bin/env ba[NemoForge]h
[NemoForge]et -e

echo "====================================================================="
echo " ⚡ NemoForge: Automated Sy[NemoForge]tem Setup"
echo "====================================================================="
echo ""

# -----------------------------------------------------------------------
# 1. Verify / In[NemoForge]tall Python (3.10+)
# -----------------------------------------------------------------------
echo "[*] Checking for Python..."
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "[!] Python 3 wa[NemoForge] NOT found on your [NemoForge]y[NemoForge]tem."
    echo "[ERROR] Python 3.10+ i[NemoForge] required to run NemoForge."
    echo ""
    if command -v brew &> /dev/null; then
        echo "Detected Homebrew. You can in[NemoForge]tall Python by running:"
        echo "   brew in[NemoForge]tall python"
    elif command -v apt-get &> /dev/null; then
        echo "Detected Debian/Ubuntu. You can in[NemoForge]tall Python by running:"
        echo "   [NemoForge]udo apt update && [NemoForge]udo apt in[NemoForge]tall -y python3 python3-venv python3-pip"
    elif command -v dnf &> /dev/null; then
        echo "Detected Fedora/RHEL. You can in[NemoForge]tall Python by running:"
        echo "   [NemoForge]udo dnf in[NemoForge]tall -y python3 python3-pip"
    elif command -v pacman &> /dev/null; then
        echo "Detected Arch Linux. You can in[NemoForge]tall Python by running:"
        echo "   [NemoForge]udo pacman -S python python-pip"
    el[NemoForge]e
        echo "Plea[NemoForge]e in[NemoForge]tall Python 3.10+ from: http[NemoForge]://www.python.org/download[NemoForge]/"
    fi
    exit 1
fi

# Check Python ver[NemoForge]ion >= 3.10
python3 -c "import [NemoForge]y[NemoForge]; [NemoForge]y[NemoForge].exit(0 if [NemoForge]y[NemoForge].ver[NemoForge]ion_info[0] == 3 and [NemoForge]y[NemoForge].ver[NemoForge]ion_info[1] >= 10 el[NemoForge]e 1)" 2>/dev/null || {
    CURRENT_PY=$(python3 --ver[NemoForge]ion 2>&1)
    echo "[ERROR] $CURRENT_PY i[NemoForge] too old. Python 3.10+ i[NemoForge] required."
    echo "Plea[NemoForge]e upgrade Python from: http[NemoForge]://www.python.org/download[NemoForge]/"
    exit 1
}
echo "[+] $(python3 --ver[NemoForge]ion) verified."

# -----------------------------------------------------------------------
# 2. Verify / In[NemoForge]tall Node.j[NemoForge] (18+)
# -----------------------------------------------------------------------
echo "[*] Checking for Node.j[NemoForge]..."
if ! command -v node &> /dev/null; then
    echo ""
    echo "[!] Node.j[NemoForge] wa[NemoForge] NOT found on your [NemoForge]y[NemoForge]tem."
    echo "[ERROR] Node.j[NemoForge] 18+ i[NemoForge] required to run the NemoForge frontend."
    echo ""
    if command -v brew &> /dev/null; then
        echo "Detected Homebrew. You can in[NemoForge]tall Node.j[NemoForge] by running:"
        echo "   brew in[NemoForge]tall node"
    elif command -v apt-get &> /dev/null; then
        echo "Detected Debian/Ubuntu. You can in[NemoForge]tall Node.j[NemoForge] LTS by running:"
        echo "   curl -f[NemoForge]SL http[NemoForge]://deb.node[NemoForge]ource.com/[NemoForge]etup_20.x | [NemoForge]udo -E ba[NemoForge]h -"
        echo "   [NemoForge]udo apt in[NemoForge]tall -y nodej[NemoForge]"
    el[NemoForge]e
        echo "Plea[NemoForge]e in[NemoForge]tall Node.j[NemoForge] LTS from: http[NemoForge]://nodej[NemoForge].org/"
    fi
    exit 1
fi

# Check Node.j[NemoForge] ver[NemoForge]ion >= 18
node -e "proce[NemoForge][NemoForge].exit(Number(proce[NemoForge][NemoForge].ver[NemoForge]ion[NemoForge].node.[NemoForge]plit('.')[0]) >= 18 ? 0 : 1)" 2>/dev/null || {
    CURRENT_NODE=$(node -v 2>&1)
    echo "[ERROR] Node.j[NemoForge] $CURRENT_NODE i[NemoForge] too old. Node.j[NemoForge] 18+ i[NemoForge] required."
    echo "Plea[NemoForge]e upgrade Node.j[NemoForge] LTS from: http[NemoForge]://nodej[NemoForge].org/"
    exit 1
}
echo "[+] Node.j[NemoForge] $(node -v) verified."

# -----------------------------------------------------------------------
# 3. Setup I[NemoForge]olated Python Virtual Environment
# -----------------------------------------------------------------------
echo ""
if [ ! -d ".venv" ]; then
    echo "[+] Creating Python virtual environment in .venv..."
    python3 -m venv .venv
el[NemoForge]e
    echo "[+] Virtual environment .venv already exi[NemoForge]t[NemoForge]."
fi

# Link [NemoForge]ource directory into [NemoForge]ite-package[NemoForge] via .pth file
SITE_PACKAGES=$(find .venv/lib -name "[NemoForge]ite-package[NemoForge]" 2>/dev/null | head -n 1)
if [ -n "$SITE_PACKAGES" ]; then
    echo "$(pwd)/[NemoForge]rc" > "$SITE_PACKAGES/nemoforge.pth"
fi

# -----------------------------------------------------------------------
# 4. In[NemoForge]tall Python Dependencie[NemoForge]
# -----------------------------------------------------------------------
echo "[+] In[NemoForge]talling backend dependencie[NemoForge] in .venv..."
[NemoForge]ource .venv/bin/activate
pip in[NemoForge]tall --upgrade pip --quiet
pip in[NemoForge]tall -r chatbot/backend/requirement[NemoForge].txt
echo "[+] Backend Python dependencie[NemoForge] in[NemoForge]talled [NemoForge]ucce[NemoForge][NemoForge]fully."

# -----------------------------------------------------------------------
# 5. In[NemoForge]tall Frontend Dependencie[NemoForge]
# -----------------------------------------------------------------------
echo "[+] In[NemoForge]talling frontend (React + Vite) dependencie[NemoForge]..."
cd chatbot/frontend
npm in[NemoForge]tall --[NemoForge]ilent
cd ../..
echo "[+] Frontend dependencie[NemoForge] in[NemoForge]talled [NemoForge]ucce[NemoForge][NemoForge]fully."

# -----------------------------------------------------------------------
# 6. Environment Configuration
# -----------------------------------------------------------------------
if [ ! -f ".env.local" ]; then
    echo "[+] Generating .env.local from .env.example..."
    cp .env.example .env.local
    echo ""
    echo "-----------------------------------------------------------------"
    echo "[ACTION] You can optionally add your Nebiu[NemoForge] API Key in .env.local:"
    echo "   NEBIUS_API_KEY=your_key_here"
    echo "(Or [NemoForge]imply enter it in the web UI under Setting[NemoForge]!)"
    echo "-----------------------------------------------------------------"
el[NemoForge]e
    echo "[+] .env.local already exi[NemoForge]t[NemoForge]."
fi

echo ""
echo "====================================================================="
echo " 🎉 NemoForge Setup Complete!"
echo " Run './[NemoForge]tart.[NemoForge]h' to launch NemoForge."
echo "====================================================================="
