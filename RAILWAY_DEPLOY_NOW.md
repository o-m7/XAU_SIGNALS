# Deploy to Railway - Quick Guide

**Status:** ✅ Ready to deploy 3 validated models
**Date:** January 7, 2026

---

## ⚡ **STEP 1: Push to GitHub (Do This First)**

Run these commands in your terminal:

```bash
# 1. Authenticate with GitHub
gh auth login

# Follow prompts:
# - Select: GitHub.com
# - Select: HTTPS
# - Select: Yes (authenticate Git)
# - Select: Login with web browser
# - Copy code, paste in browser, approve

# 2. Push to GitHub
git push origin main
```

**✅ Done? Proceed to Step 2**

---

## ⚡ **STEP 2: Deploy to Railway**

### **A. Install Railway CLI (if not installed)**

```bash
npm install -g @railway/cli
```

### **B. Login and Initialize**

```bash
# Login to Railway
railway login

# Link to existing project OR create new one
railway link
# OR
railway init
```

### **C. Set Environment Variables**

```bash
# Required variables
railway variables set POLYGON_API_KEY="your_polygon_key_here"
railway variables set TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
railway variables set TELEGRAM_CHAT_ID="your_telegram_chat_id"

# Optional (already have good defaults)
railway variables set WS_MODE="all"
railway variables set BASE_SYMBOL="XAU"
railway variables set QUOTE_SYMBOL="USD"
```

**Get your values:**
- **Polygon API Key:** https://polygon.io/dashboard/api-keys
- **Telegram Bot Token:** Talk to @BotFather on Telegram
- **Telegram Chat ID:** Forward a message to @userinfobot

### **D. Deploy**

```bash
# Deploy to Railway
railway up

# This will:
# - Push your code to Railway
# - Build the Docker container
# - Start running all 3 models
# - Begin sending signals to Telegram
```

### **E. Monitor**

```bash
# Watch logs in real-time
railway logs

# Check status
railway status
```

---

## 📊 **What Will Happen**

When deployed, Railway will:

1. **Install dependencies** (from requirements.txt)
2. **Load 3 models:**
   - model3_cmf_macd_v4 (PRIMARY - 61.7% WR, 16.8 trades/day)
   - model1_high_conf (SECONDARY - 61.3% WR, 3.1 trades/day)
   - model_rf_v4 (TERTIARY - 57.7% WR, 7.0 trades/day)

3. **Connect to Polygon.io** WebSocket
4. **Backfill historical data** (15min bars)
5. **Start generating signals** (~27/day total)
6. **Send to Telegram** (each labeled by model)

---

## ✅ **Verify Deployment**

Check Railway logs for these messages:

```
✓ Loaded model3_cmf_macd_v4 from ...
✓ Loaded model1_high_conf from ...
✓ Loaded model_rf_v4 from ...
MultiModelSignalEngine initialized with 3 models
WebSocket connected to Polygon.io
Backfill complete: 100 bars loaded
```

Check Telegram for connection test message.

---

## 🔧 **Troubleshooting**

### **Issue: Build fails**

```bash
# Check logs
railway logs

# Redeploy
railway up --detach
```

### **Issue: Models not loading**

Check Railway logs:
```bash
railway logs | grep -i error
```

Verify model files exist in repo:
```bash
git ls-files models/*.joblib
```

### **Issue: No signals received**

1. Check Railway is running: `railway status`
2. Check logs: `railway logs`
3. Verify Telegram env vars: `railway variables`
4. Check Polygon connection in logs

---

## 💰 **Expected Costs**

**Railway:**
- Free tier: $5/month credit (should be enough)
- Paid tier: ~$5-10/month for 24/7 operation

**APIs:**
- Polygon.io: Free tier (real-time data)
- Telegram: Free

---

## 🛑 **Stop/Restart**

```bash
# View deployments
railway status

# Restart
railway restart

# Stop (not recommended while trading)
railway down
```

---

## 📈 **Monitoring**

**Daily:**
- Check Telegram for signals
- Verify all 3 models sending signals
- Track win rate in spreadsheet

**Weekly:**
- Review Railway logs
- Check uptime
- Monitor performance vs backtest

---

## 🚀 **You're Ready!**

**Current status:**
- ✅ 3 models validated (60%+ WR on 2025 OOS)
- ✅ Code committed to git
- ⏳ Need to push to GitHub
- ⏳ Need to deploy to Railway

**Next command:** `gh auth login` then `git push origin main`

---

**Questions?** Check `DEPLOYMENT_GUIDE.md` for detailed instructions.
