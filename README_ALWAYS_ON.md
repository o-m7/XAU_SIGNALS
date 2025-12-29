# Running Signal Engine with Laptop Closed

## Quick Setup

Run the setup script:
```bash
./setup_always_on.sh
```

## Manual Setup

### 1. Configure macOS Power Settings

**Option A: System Settings (GUI)**
1. Open **System Settings** → **Battery**
2. Enable "Prevent automatic sleeping when display is off"
3. Set "Turn display off after" to a reasonable time (e.g., 10 minutes)

**Option B: Terminal (requires admin)**
```bash
sudo pmset -a sleep 0 displaysleep 10
sudo pmset -a disablesleep 1
```

### 2. Keep Laptop Plugged In

⚠️ **CRITICAL**: macOS will only stay awake with lid closed if:
- Laptop is plugged into AC power
- Power settings are configured correctly

If unplugged, the laptop will sleep (macOS safety feature).

### 3. Start the Signal Engine

```bash
./start_signal_engine.sh start
```

The script uses `caffeinate` to prevent sleep even when the lid is closed.

### 4. (Optional) Auto-Start on Boot

Install the launchd service:
```bash
cp com.xauusd.signals.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.xauusd.signals.plist
```

## Verify It's Running

```bash
./start_signal_engine.sh status
./start_signal_engine.sh logs
```

## Alternative: Cloud Deployment

For true 24/7 operation without laptop dependency, consider deploying to:
- **Railway** (already configured - see `railway.json`)
- **Heroku**
- **AWS EC2**
- **DigitalOcean Droplet**

## Troubleshooting

**Laptop sleeps when lid closed:**
- Check if laptop is plugged in
- Verify power settings: `pmset -g`
- Try: `sudo pmset -a sleep 0`

**Signal engine stops:**
- Check logs: `./start_signal_engine.sh logs`
- Check if process is running: `ps aux | grep live_runner`
- Restart: `./start_signal_engine.sh restart`
