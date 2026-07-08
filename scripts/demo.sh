#!/bin/bash
# BioDynamics Agent — One-command demo
# Runs one pathway end-to-end: starts backend + frontend, opens browser
set -e

# Configuration
PATHWAY="${1:-egfr}"  # Default to EGFR, accept pathway class as arg
BACKEND_PORT=8000
FRONTEND_PORT=3000

echo "🧬 BioDynamics Agent — Demo Mode"
echo "   Pathway: $PATHWAY"
echo ""

# Check prerequisites
command -v python >/dev/null 2>&1 || { echo "❌ Python not found"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "❌ Node not found"; exit 1; }

# Start backend
echo "▶ Starting backend (port $BACKEND_PORT)..."
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT &
BACKEND_PID=$!
cd ..

# Wait for backend
echo "⏳ Waiting for backend..."
sleep 5

# Start frontend
echo "▶ Starting frontend (port $FRONTEND_PORT)..."
cd frontend
npm run dev -- -p $FRONTEND_PORT &
FRONTEND_PID=$!
cd ..

# Wait for frontend
echo "⏳ Waiting for frontend..."
sleep 8

# Open browser
echo "🌐 Opening browser..."
URL="http://localhost:$FRONTEND_PORT/workspace?pathway=$PATHWAY"
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL"
elif command -v open >/dev/null 2>&1; then
    open "$URL"
fi

echo ""
echo "✅ Demo running!"
echo "   Frontend: http://localhost:$FRONTEND_PORT"
echo "   Backend:  http://localhost:$BACKEND_PORT/docs"
echo "   Workspace: $URL"
echo ""
echo "Press Ctrl+C to stop..."

# Trap Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
