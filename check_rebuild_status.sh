#!/bin/bash
# Check rebuild training status

echo "Checking rebuild status..."
echo ""

# Check if process is running
if ps aux | grep -q "[p]ython rebuild_from_scratch.py"; then
    echo "✅ Training is RUNNING"
    echo ""

    # Show latest output
    if [ -f "rebuild_output.log" ]; then
        echo "Latest output (last 30 lines):"
        echo "================================"
        tail -30 rebuild_output.log
    else
        echo "No log file found yet"
    fi
else
    echo "❌ Training is NOT running"
    echo ""

    # Check if completed
    if [ -d "models/rebuilt_from_scratch" ]; then
        echo "✅ Found rebuilt models directory"
        ls -lh models/rebuilt_from_scratch/
    else
        echo "⚠️ No rebuilt models found"
    fi

    # Show full output if available
    if [ -f "rebuild_output.log" ]; then
        echo ""
        echo "Full training log:"
        echo "=================="
        cat rebuild_output.log
    fi
fi
