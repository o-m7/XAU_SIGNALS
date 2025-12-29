# Deploy Signal Engine to Cloud (24/7 Operation)

## Problem
- Local laptop must stay awake for signal engine to run
- If laptop sleeps, process stops
- Need 24/7 operation without keeping laptop awake

## Solution: Deploy to Railway (or similar cloud service)

### Option 1: Railway (Recommended - Already Configured)

1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Deploy:**
   ```bash
   cd /Users/omar/Desktop/ML/xauusd_signals
   railway init
   railway up
   ```

3. **Set Environment Variables in Railway Dashboard:**
   - `POLYGON_API_KEY` - Your Polygon API key
   - `TELEGRAM_BOT_TOKEN` - Your Telegram bot token
   - `TELEGRAM_CHAT_ID` - Your Telegram chat ID
   - `BASE_SYMBOL` - XAU (default)
   - `QUOTE_SYMBOL` - USD (default)
   - **`WS_MODE`** - **MUST be set to `all`** (connects to all 3 channels: quotes, minute aggregates, second aggregates)
     - ⚠️ **CRITICAL**: If `WS_MODE` is not set or set to anything other than `all`, the system will only connect to 1 channel (quotes)
     - Check logs for: `🔧 Environment WS_MODE:` to verify it's being read correctly

4. **Monitor:**
   ```bash
   railway logs
   ```

### Option 2: Remove Caffeinate (Allow Laptop to Sleep)

If you want the laptop to sleep normally but keep process running when awake:

**Edit `start_signal_engine.sh`:**
```bash
# Remove or comment out caffeinate line
# caffeinate -w $$ &
```

**Note:** Process will stop when laptop sleeps. For true 24/7, use cloud deployment.

### Option 3: Use a VPS/Server

Deploy to:
- DigitalOcean Droplet
- AWS EC2
- Google Cloud Compute
- Any Linux server

Then run:
```bash
./start_signal_engine.sh start
```

## Current Status

✅ **Local Setup:**
- Signal engine running (PID: 82898)
- WebSocket connected (3 channels)
- 4,824+ events received
- Warnings fixed
- Logging optimized

⚠️ **Limitation:**
- Laptop must stay awake for local operation
- If laptop sleeps, process stops
- For 24/7: Deploy to Railway/cloud

## Quick Fix for Local: Remove Aggressive Caffeinate

The current setup uses `caffeinate -w $$` which only prevents sleep while the process runs. This is better than before, but:

- **If you stop the signal engine:** Laptop can sleep normally
- **If signal engine is running:** Laptop stays awake (prevents sleep)

To allow laptop to sleep even while running (process will stop when laptop sleeps):
- Remove caffeinate from `start_signal_engine.sh`
- Or deploy to cloud for true 24/7 operation

