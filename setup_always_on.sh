#!/bin/bash
#
# Setup script to make signal engine run even when laptop lid is closed
#
# This script:
# 1. Configures macOS to not sleep when lid is closed (requires admin)
# 2. Sets up launchd service for auto-start on boot
# 3. Provides instructions for manual setup
#

cd /Users/omar/Desktop/ML/xauusd_signals

echo "=========================================="
echo "XAUUSD Signal Engine - Always-On Setup"
echo "=========================================="
echo ""

# Check if running as admin
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  Running as root. This is not recommended."
    exit 1
fi

echo "Step 1: Configure macOS System Settings"
echo "--------------------------------------"
echo ""
echo "To allow the laptop to run with lid closed:"
echo "1. Open System Settings → Battery"
echo "2. Set 'Prevent automatic sleeping when display is off' to ON"
echo "3. Or use: sudo pmset -a sleep 0 displaysleep 10"
echo ""
read -p "Do you want to configure this now? (requires admin password) [y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Configuring power settings..."
    sudo pmset -a sleep 0 displaysleep 10
    sudo pmset -a disablesleep 1
    echo "✅ Power settings configured"
else
    echo "⏭️  Skipping power settings (you can do this manually)"
fi

echo ""
echo "Step 2: Install Launch Agent (Auto-start on boot)"
echo "--------------------------------------"
echo ""
read -p "Install launchd service for auto-start? [y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
    mkdir -p "$LAUNCH_AGENTS_DIR"
    
    cp com.xauusd.signals.plist "$LAUNCH_AGENTS_DIR/"
    
    # Load the service
    launchctl unload "$LAUNCH_AGENTS_DIR/com.xauusd.signals.plist" 2>/dev/null
    launchctl load "$LAUNCH_AGENTS_DIR/com.xauusd.signals.plist"
    
    echo "✅ Launch agent installed and loaded"
    echo "   The signal engine will start automatically on boot"
else
    echo "⏭️  Skipping launch agent installation"
fi

echo ""
echo "Step 3: Important Notes"
echo "--------------------------------------"
echo ""
echo "⚠️  CRITICAL REQUIREMENTS:"
echo "   1. Laptop MUST be plugged into power (AC adapter)"
echo "   2. macOS will NOT sleep when lid is closed IF plugged in"
echo "   3. If unplugged, laptop will sleep (this is a macOS limitation)"
echo ""
echo "✅ Current Status:"
echo "   - Signal engine: $(./start_signal_engine.sh status 2>&1 | head -1)"
echo ""
echo "📝 To start manually: ./start_signal_engine.sh start"
echo "📝 To stop: ./start_signal_engine.sh stop"
echo "📝 To view logs: ./start_signal_engine.sh logs"
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="

