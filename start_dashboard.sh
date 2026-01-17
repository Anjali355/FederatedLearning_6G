#!/bin/bash

# Interactive Dashboard Quick Start Script
# This script starts the dashboard and training simultaneously

echo "🚀 Starting Interactive Federated Learning Dashboard..."
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Activating virtual environment..."
    source .venv/bin/activate
fi

# Start the dashboard in the background
echo "📊 Starting dashboard server at http://localhost:8050"
python interactive_dashboard.py > dashboard.log 2>&1 &
DASHBOARD_PID=$!

# Wait for dashboard to start
sleep 3

# Open browser (macOS)
if command -v open &> /dev/null; then
    echo "🌐 Opening browser..."
    open http://localhost:8050
fi

# Show instructions
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Dashboard is running!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📍 Dashboard URL: http://localhost:8050"
echo "🔄 Auto-updates every 2 seconds"
echo ""
echo "Now run your training in another terminal:"
echo ""
echo "  python main.py --data synthetic_network_dataset.csv \\"
echo "                 --num-clients 8 --rounds 10 --cpu"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📝 Dashboard logs: dashboard.log"
echo "🛑 To stop dashboard: kill $DASHBOARD_PID"
echo "   Or use: pkill -f interactive_dashboard.py"
echo ""

# Keep script running and show log
tail -f dashboard.log
