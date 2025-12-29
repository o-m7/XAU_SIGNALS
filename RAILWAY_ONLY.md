# Railway-Only Operation

## ⚠️ Important: Only Run One Instance

Polygon.io has a connection limit per account. If you run the signal engine both locally AND on Railway, you will hit the "Maximum number of websocket connections exceeded" error.

## ✅ Solution: Railway-Only Operation

### 1. Stop Local Instance

```bash
cd /Users/omar/Desktop/ML/xauusd_signals
./start_signal_engine.sh stop
```

### 2. Verify Local is Stopped

```bash
./start_signal_engine.sh status
# Should show: "Signal engine not running"
```

### 3. Check for Any Running Processes

```bash
ps aux | grep -i "live_runner\|python.*live" | grep -v grep
# Should show nothing (no processes)
```

### 4. Railway Configuration

Make sure Railway has these environment variables set:

- `POLYGON_API_KEY` - Your Polygon API key
- `TELEGRAM_BOT_TOKEN` - Your Telegram bot token
- `TELEGRAM_CHAT_ID` - Your Telegram chat ID
- **`WS_MODE`** - **MUST be `all`** (for all 3 channels)

### 5. Monitor Railway Logs

```bash
railway logs
```

Look for:
- `✅ WS Mode: ALL (3 channels: quotes, minute, second)`
- `🎯 Target channels set: 3 channels: C.XAU/USD, CA.XAU/USD, CAS.XAU/USD`
- `✅ Successfully connected to all 3 channels`

## 🚫 Don't Run Locally While Railway is Running

If you see this error:
```
Status: max_connections - Maximum number of websocket connections exceeded
```

It means:
- You have multiple instances running (local + Railway)
- Or you have multiple Railway deployments
- **Solution:** Stop all instances except one

## 🔄 To Switch Back to Local

1. Stop Railway deployment
2. Start local: `./start_signal_engine.sh start`
3. Verify: `./start_signal_engine.sh status`

## 📊 Current Status

- ✅ Local instance stopped
- ✅ Railway should be the only running instance
- ✅ Connection limit error should be resolved

