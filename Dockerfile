# =============================================================================
# NemoForge Autonomous SWE System Container
# Multi-Platform Runtime (Linux, macOS, Windows via Docker Desktop)
# Bundles: Python 3.13 + OpenJDK 17 + Maven 3.9 + Node.js 20 + Playwright
# Python dependencies are installed from the locked set in uv.lock via uv.
# =============================================================================

FROM python:3.13-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONPATH="/app/src:/app"
# uv: install into /app/.venv, copy (not hardlink) from the cache, never download a Python
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 1. Install System Dependencies: OpenJDK 17, Maven, Node.js 20, Git, Curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    openjdk-17-jdk-headless \
    maven \
    ca-certificates \
    gnupg \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Node.js 20 LTS and Playwright System Prerequisites
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y nodejs \
    && npm install -g @playwright/mcp @modelcontextprotocol/server-filesystem \
    && npx playwright install-deps chromium \
    && npx playwright install chromium \
    && rm -rf /var/lib/apt/lists/*

# 3. Resolve JAVA_HOME from the installed JDK (architecture independent)
RUN ln -s $(dirname $(dirname $(readlink -f $(which javac)))) /opt/java
ENV JAVA_HOME=/opt/java
ENV PATH="/app/.venv/bin:/opt/java/bin:${PATH}"

# 4. Install uv and sync the locked Python dependencies (no dev group)
COPY --from=ghcr.io/astral-sh/uv:0.9.13 /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 5. Copy NemoForge Source Code & Assets
COPY src/ /app/src/
COPY chatbot/ /app/chatbot/
COPY brain/ /app/brain/

# 6. Build Frontend Distribution
WORKDIR /app/chatbot/frontend
RUN npm install --legacy-peer-deps && npm run build
WORKDIR /app

# 7. Expose Ports (8002: Backend API, 3000: Frontend UI)
EXPOSE 8002
EXPOSE 3000

# 8. Default Entrypoint: Launch Backend with uvicorn (from /app/.venv via PATH)
CMD ["uvicorn", "chatbot.backend.main:app", "--host", "0.0.0.0", "--port", "8002"]
