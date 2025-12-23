#!/bin/bash
#
# Setup Weekly Model Retraining (Saturday 6 AM)
#
# This script sets up a cron job to retrain the model every Saturday
# with the freshest data and trade feedback.
#

PROJECT_DIR="/Users/omar/Desktop/ML/xauusd_signals"
PYTHON="$PROJECT_DIR/venv/bin/python"
SCRIPT="$PROJECT_DIR/src/retrain_weekly.py"
LOG="$PROJECT_DIR/logs/retrain.log"

# Ensure logs directory exists
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/data/feedback"

# Create the cron entry
CRON_ENTRY="0 6 * * 6 cd $PROJECT_DIR && $PYTHON $SCRIPT >> $LOG 2>&1"

# Check if cron entry already exists
if crontab -l 2>/dev/null | grep -q "retrain_weekly.py"; then
    echo "⚠️  Weekly retrain cron job already exists"
    echo "Current cron jobs:"
    crontab -l | grep retrain
    echo ""
    read -p "Do you want to replace it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove existing and add new
        (crontab -l 2>/dev/null | grep -v "retrain_weekly.py"; echo "$CRON_ENTRY") | crontab -
        echo "✅ Cron job updated"
    else
        echo "Keeping existing cron job"
        exit 0
    fi
else
    # Add new cron entry
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
    echo "✅ Weekly retrain cron job added"
fi

echo ""
echo "📅 Schedule: Every Saturday at 6:00 AM"
echo "📁 Log file: $LOG"
echo ""
echo "Current cron jobs:"
crontab -l | grep -E "retrain|xauusd" || echo "(none)"
echo ""
echo "To run manually:"
echo "  cd $PROJECT_DIR && source venv/bin/activate && python src/retrain_weekly.py"
echo ""
echo "To check logs:"
echo "  tail -f $LOG"

