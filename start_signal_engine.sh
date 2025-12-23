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
CAFFEINE_PIDFILE=".caffeinate.pid"

start() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Signal engine already running (PID: $PID)"
            exit 1
        else
            echo "Removing stale PID file..."
            rm -f "$PIDFILE" "$LOCKFILE" "$CAFFEINE_PIDFILE"
        fi
    fi
    
    echo "Starting XAUUSD Signal Engine..."
    
    # Prevent Mac from sleeping (keeps running when lid closed if plugged in)
    caffeinate -s -w $$ &
    CAFFEINE_PID=$!
    echo $CAFFEINE_PID > "$CAFFEINE_PIDFILE"
    echo "☕ Caffeinate started (PID: $CAFFEINE_PID) - Mac will stay awake"
    
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
        echo ""
        echo "⚠️  IMPORTANT: Keep laptop plugged in to prevent sleep!"
    else
        echo "❌ Failed to start signal engine"
        rm -f "$PIDFILE" "$CAFFEINE_PIDFILE"
        # Kill caffeinate if engine failed
        kill $CAFFEINE_PID 2>/dev/null
        exit 1
    fi
}

stop() {
    # Stop signal engine
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
        rm -f "$LOCKFILE"
    fi
    
    # Stop caffeinate
    if [ -f "$CAFFEINE_PIDFILE" ]; then
        CAFF_PID=$(cat "$CAFFEINE_PIDFILE")
        if ps -p $CAFF_PID > /dev/null 2>&1; then
            kill $CAFF_PID 2>/dev/null
            echo "☕ Caffeinate stopped"
        fi
        rm -f "$CAFFEINE_PIDFILE"
    fi
    # Also kill any orphaned caffeinate processes from this script
    pkill -f "caffeinate -s" 2>/dev/null
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

