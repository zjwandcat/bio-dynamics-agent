# =============================================================================
# BioDynamics Agent v4 - Multi-stage Dockerfile
# Task G.3: Docker containerization
# =============================================================================
# Three stages:
#   1. backend          — FastAPI + uvicorn (Python 3.11-slim)
#   2. frontend-build   — Next.js production build (Node 20-slim)
#   3. frontend         — Next.js production runtime (Node 20-slim)
#
# Build context: project root (bio-dynamics-agent/).
# Build a single stage via `docker build --target <stage> .` or use
# docker-compose.yml which wires both backend + frontend services.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Backend (FastAPI / uvicorn)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS backend

# System-level build deps for compiling C extensions in numpy / scipy /
# pillow / chromadb / sentence-transformers (torch wheels are prebuilt
# but a few transitive packages still need gcc).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libssl-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Install Python deps first to maximize Docker layer caching.
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source (app/, scripts/, benchmarks/, etc.).
COPY backend/ ./

# Persist ChromaDB / sandbox logs / metrics outside the container via volume
# (see docker-compose.yml → ./backend/data:/app/backend/data).
EXPOSE 8000

# uvicorn entrypoint — app.main:app, bind all interfaces.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# -----------------------------------------------------------------------------
# Stage 2: Frontend build (Next.js production build)
# -----------------------------------------------------------------------------
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend

# Install deps first (package.json + package-lock.json) for layer caching.
# --legacy-peer-deps avoids peer-dep conflicts with Next.js 16 / React 19.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --legacy-peer-deps

# Copy the rest of the frontend source + all config files needed by `next build`
# (next.config.ts, tsconfig.json, postcss.config.mjs, components.json, etc.).
COPY frontend/ ./

RUN npm run build


# -----------------------------------------------------------------------------
# Stage 3: Frontend production runtime
# -----------------------------------------------------------------------------
FROM node:20-slim AS frontend

WORKDIR /app/frontend

ENV NODE_ENV=production

# Copy build output + runtime deps + config + static assets.
# next.config.ts is read by `next start` at boot, so it must be present.
COPY --from=frontend-build /app/frontend/.next ./.next
COPY --from=frontend-build /app/frontend/node_modules ./node_modules
COPY --from=frontend-build /app/frontend/package.json ./
COPY --from=frontend-build /app/frontend/next.config.ts ./
COPY --from=frontend-build /app/frontend/tsconfig.json ./
COPY --from=frontend-build /app/frontend/postcss.config.mjs ./
COPY --from=frontend-build /app/frontend/components.json ./
COPY --from=frontend-build /app/frontend/public ./public

EXPOSE 3000

CMD ["npm", "start"]
