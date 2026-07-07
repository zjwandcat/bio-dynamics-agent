#!/bin/bash
# BioDynamics Agent — Run all 10 benchmarks
set -e

BACKEND_PORT=8000

echo "🧪 BioDynamics Agent — Benchmark Suite"
echo ""

# Check prerequisites
command -v python >/dev/null 2>&1 || { echo "❌ Python not found"; exit 1; }

# Check if backend is running
if ! curl -s "http://localhost:$BACKEND_PORT/health" >/dev/null 2>&1; then
    echo "▶ Starting backend (port $BACKEND_PORT)..."
    cd backend
    python -m uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT &
    BACKEND_PID=$!
    cd ..
    echo "⏳ Waiting for backend..."
    sleep 5
    STARTED_BACKEND=1
else
    echo "✅ Backend already running"
    STARTED_BACKEND=0
fi

# Run benchmarks via SSE endpoint
echo ""
echo "▶ Running 10-pathway benchmark suite..."
echo ""

# Use curl to stream SSE events
curl -N -X POST "http://localhost:$BACKEND_PORT/api/v4/benchmarks/run" \
    -H "Content-Type: application/json" \
    -d '{"pathway_classes": ["all"]}' 2>/dev/null | while IFS= read -r line; do
    # Parse SSE events
    if [[ "$line" == data:* ]]; then
        echo "$line"
    fi
done

echo ""
echo "✅ Benchmark suite complete!"
echo "   Results: http://localhost:3000/benchmarks"

# Cleanup
if [ "$STARTED_BACKEND" = "1" ]; then
    echo "Stopping backend..."
    kill $BACKEND_PID 2>/dev/null
fi
