#!/bin/bash

# Start MLflow Gateway UI alongside MLflow
# This script starts both servers in the background

cd "$(dirname "$0")"

echo "======================================================================"
echo "🚀 Starting MLflow + Gateway UI"
echo "======================================================================"

# Check if MLflow is already running
if lsof -i:5000 >/dev/null 2>&1; then
    echo "✅ MLflow UI already running on port 5000"
else
    echo "🔄 Starting MLflow UI on port 5000..."
    venv311/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 > /tmp/mlflow_ui.log 2>&1 &
    MLFLOW_PID=$!
    echo "   PID: $MLFLOW_PID"
    sleep 2
fi

# Check if Gateway UI is already running
if lsof -i:5001 >/dev/null 2>&1; then
    echo "✅ Gateway UI already running on port 5001"
else
    echo "🔄 Starting Gateway UI on port 5001..."
    venv311/bin/python mlflowlite/ui/gateway_server.py > /tmp/gateway_ui.log 2>&1 &
    GATEWAY_PID=$!
    echo "   PID: $GATEWAY_PID"
    sleep 2
fi

echo ""
echo "======================================================================"
echo "✅ Servers started!"
echo "======================================================================"
echo ""
echo "📊 MLflow UI:   http://localhost:5000"
echo "🚀 Gateway UI:  http://localhost:5001/gateway"
echo ""
echo "📝 Logs:"
echo "   MLflow:  tail -f /tmp/mlflow_ui.log"
echo "   Gateway: tail -f /tmp/gateway_ui.log"
echo ""
echo "🛑 To stop servers:"
echo "   pkill -f mlflow"
echo "   pkill -f gateway_server"
echo "======================================================================"

