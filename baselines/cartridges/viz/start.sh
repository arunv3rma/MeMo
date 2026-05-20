#!/bin/bash

# Start the dataset visualization application

echo "Starting Dataset Visualization App..."

# Default configuration (can be overridden by environment variables)
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-"8001"}
export RELOAD=${RELOAD:-"true"}
export CORS_ENABLED=${CORS_ENABLED:-"true"}
export CORS_ORIGINS=${CORS_ORIGINS:-"http://localhost:5173"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            export HOST="$2"
            shift 2
            ;;
        --port)
            export PORT="$2"
            shift 2
            ;;
        --no-reload)
            export RELOAD="false"
            shift
            ;;
        --no-cors)
            export CORS_ENABLED="false"
            shift
            ;;
        --cors-origins)
            export CORS_ORIGINS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --host HOST          Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT          Port to bind to (default: 8001)"
            echo "  --no-reload          Disable auto-reload"
            echo "  --no-cors            Disable CORS"
            echo "  --cors-origins URLS  Comma-separated CORS origins"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if Python backend dependencies are installed
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip3 install --upgrade pip setuptools wheel
    pip3 install -r requirements.txt
fi

# Start the Python backend server
echo "Starting FastAPI server on $HOST:$PORT..."
echo "Configuration:"
echo "  - CORS enabled: $CORS_ENABLED"
echo "  - CORS origins: $CORS_ORIGINS"
echo "  - Auto-reload: $RELOAD"

python3 src/server.py --host "$HOST" --port "$PORT" $([ "$RELOAD" = "true" ] && echo "--reload") $([ "$CORS_ENABLED" = "false" ] && echo "--no-cors") --cors-origins "$CORS_ORIGINS" &
BACKEND_PID=$!

# Wait a moment for the backend to start
sleep 2

# Start the Vite development server
echo "Starting Vite development server..."
npm run dev &
FRONTEND_PID=$!

# Function to cleanup processes on exit
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "
Dataset Visualization App is running!
- Frontend: http://localhost:5173

Press Ctrl+C to stop both servers.
"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID