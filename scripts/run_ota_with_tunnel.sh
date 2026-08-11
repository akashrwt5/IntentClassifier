#!/bin/bash

# Configuration
PORT=8000
SUBDOMAIN="intent-classifier-ota"

echo "=================================================="
echo "🚀 Starting Dev OTA Server in the background..."
echo "=================================================="

# Start the Python server in the background
.venv/bin/python scripts/dev_ota_server.py &
SERVER_PID=$!

# Ensure the Python server is killed when this script exits (Ctrl+C)
trap "echo 'Shutting down server...'; kill $SERVER_PID" EXIT

# Give the server a few seconds to boot up completely
sleep 3

echo ""
echo "=================================================="
echo "🌐 Starting Localtunnel on https://$SUBDOMAIN.loca.lt"
echo "=================================================="
echo "⚠️ IMPORTANT: When you visit the URL for the first time,"
echo "it might ask for an 'Endpoint IP'. Just click 'Click to Continue' or"
echo "enter your public IP if prompted (this is a localtunnel security feature)."
echo "=================================================="

# Run localtunnel in the foreground
npx localtunnel --port $PORT --subdomain $SUBDOMAIN
