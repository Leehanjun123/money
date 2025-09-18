#!/bin/bash
# Railway startup script

echo "Starting Style Mate Production Server..."
echo "PORT: ${PORT:-8000}"
echo "ENVIRONMENT: ${ENVIRONMENT:-production}"

# Start the server with the PORT from environment
exec uvicorn main_production:app --host 0.0.0.0 --port ${PORT:-8000}