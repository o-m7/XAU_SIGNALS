#!/bin/bash
#
# XAUUSD Signal Engine Startup Script
# 
# Usage:
#   ./start_signal_engine.sh        # Start the engine
#   ./start_signal_engine.sh stop   # Stop the engine
#   ./start_signal_engine.sh status # Check status
#   ./start_signal_engine.sh logs   # View logs
#

cd /Users/omar/Desktop/ML/xauusd_signals
source venv/bin/activate

PIDFILE=".live_runner.pid"
LOCKFILE=".live_runner.lock"
LOGFILE="live_runner.log"

start() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Signal engine already running (PID: $PID)"
            exit 1
        else
            echo "Removing stale PID file..."
            rm -f "$PIDFILE" "$LOCKFILE"
        fi
    fi
    
    echo "Starting XAUUSD Signal Engine..."
    # Set WS_MODE=all to connect to all 3 WebSocket channels
    export WS_MODE=all
    nohup python -m src.live.live_runner \
        --backfill \
        --threshold_long 0.70 \
        --threshold_short 0.25 \
        >> "$LOGFILE" 2>&1 &
    
    echo $! > "$PIDFILE"
    sleep 2
    
    if ps -p $(cat "$PIDFILE") > /dev/null 2>&1; then
        echo "✅ Signal engine started (PID: $(cat $PIDFILE))"
        echo "   Logs: tail -f $LOGFILE"
    else
        echo "❌ Failed to start signal engine"
        rm -f "$PIDFILE"
        exit 1
    fi
}

stop() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Stopping signal engine (PID: $PID)..."
            kill $PID
            sleep 2
            if ps -p $PID > /dev/null 2>&1; then
                echo "Force killing..."
                kill -9 $PID
            fi
            rm -f "$PIDFILE" "$LOCKFILE"
            echo "✅ Signal engine stopped"
        else
            echo "Process not running, cleaning up..."
            rm -f "$PIDFILE" "$LOCKFILE"
        fi
    else
        echo "Signal engine not running (no PID file)"
        # Clean up lock file just in case
        rm -f "$LOCKFILE"
    fi
}

status() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "✅ Signal engine running (PID: $PID)"
            echo ""
            echo "Recent activity:"
            tail -5 "$LOGFILE"
        else
            echo "❌ Signal engine not running (stale PID file)"
        fi
    else
        echo "❌ Signal engine not running"
    fi
}

logs() {
    tail -f "$LOGFILE"
}

case "${1:-start}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 1
        start
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac

