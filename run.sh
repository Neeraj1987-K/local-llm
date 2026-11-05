#!/usr/bin/env bash
# Start the API server.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
exec uvicorn api:app --host 0.0.0.0 --port 8000