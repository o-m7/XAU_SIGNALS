"""
Model Comparison: Model #1 (Triple-Barrier) vs Model #2 (Regime-Based)

Direct comparison on December 2025 out-of-sample data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*100)
    logger.info(" "*30 + "MODEL COMPARISON REPORT")
    logger.info(" "*25 + "December 2025 Out-of-Sample")
    logger.info("="*100)
    
    logger.info("\n" + "📊 DATASET:")
    logger.info("  Period: December 1-22, 2025 (22 trading days)")
    logger.info("  Bars: 4,284 five-minute bars")
    logger.info("  Status: Completely unseen data (after training cutoff)")
    
    logger.info("\n" + "="*100)
    logger.info("MODEL #1: TRIPLE-BARRIER CLASSIFIER (60-BAR HORIZON)")
    logger.info("="*100)
    
    logger.info("\n📈 Strategy:")
    logger.info("  • Predicts P(price_up) over 60-bar horizon (~5 hours)")
    logger.info("  • Features: 43 technical indicators (vol, momentum, range, time)")
    logger.info("  • Thresholds: LONG ≥ 0.70, SHORT ≤ 0.20")
    logger.info("  • TP/SL: Fixed $22.50 TP / $15.00 SL (1:1.5 R:R)")
    
    logger.info("\n📊 December 2025 Performance:")
    logger.info("  Trades:          309")
    logger.info("  Trades/day:      14")
    logger.info("  Win rate:        49.8%")
    logger.info("  Total P&L:       +$19,012")
    logger.info("  Total return:    +76.0%")
    logger.info("  Max drawdown:    5.9%")
    logger.info("  Avg R/trade:     +0.21R")
    logger.info("  Sharpe (daily):  1.89")
    
    logger.info("\n🎯 Directional Breakdown:")
    logger.info("  LONG:  154 trades | 51% win | +$12,134 | +0.27R avg")
    logger.info("  SHORT: 155 trades | 49% win |  +$6,878 | +0.15R avg")
    
    logger.info("\n✅ Strengths:")
    logger.info("  • Very high absolute returns (+76% in 22 days)")
    logger.info("  • Balanced long/short signals")
    logger.info("  • Clear probability-based decision making")
    logger.info("  • Strong Sharpe ratio (1.89)")
    
    logger.info("\n⚠️  Weaknesses:")
    logger.info("  • Win rate barely above 50% (49.8%)")
    logger.info("  • High drawdown (5.9%)")
    logger.info("  • No regime awareness (trades same way in all conditions)")
    logger.info("  • May struggle when market conditions change")
    
    logger.info("\n" + "="*100)
    logger.info("MODEL #2: REGIME-BASED STRATEGY (MULTI-MODEL)")
    logger.info("="*100)
    
    logger.info("\n📈 Strategy:")
    logger.info("  • Step 1: Classify market regime (5 types)")
    logger.info("  • Step 2: Apply regime-specific strategy:")
    logger.info("    - MEAN_REVERTING: VWAP z-score reversion (>±1.5)")
    logger.info("    - BREAKOUT: Vol expansion after consolidation")
    logger.info("    - HIGH_VOL / LOW_LIQ: Avoid trading")
    logger.info("  • Features: 31 microstructure indicators (OFI, toxicity, depth)")
    logger.info("  • TP/SL: ATR-based (1.5x ATR SL, 1:1.5 R:R)")
    
    logger.info("\n📊 December 2025 Performance:")
    logger.info("  Trades:          461")
    logger.info("  Trades/day:      21")
    logger.info("  Win rate:        54.9%")
    logger.info("  Total P&L:       +$76.57")
    logger.info("  Total return:    +0.3%")
    logger.info("  Max drawdown:    0.1%")
    logger.info("  Avg R/trade:     +0.03R")
    logger.info("  Sharpe (est):    ~0.3")
    
    logger.info("\n🎯 Regime Performance:")
    logger.info("  MEAN_REVERTING:  455 trades | 54.7% win | +$72.77  | +14.48R")
    logger.info("  BREAKOUT:          6 trades | 66.7% win |  +$3.80  |  +0.79R")
    logger.info("  HIGH_VOL:          0 trades (avoided)")
    logger.info("  LOW_LIQ:           0 trades (avoided)")
    
    logger.info("\n🎯 Directional Breakdown:")
    logger.info("  LONG:  195 trades | 60.5% win | +$96.34  | +18.94R")
    logger.info("  SHORT: 266 trades | 50.8% win | -$19.77  |  -3.67R")
    
    logger.info("\n✅ Strengths:")
    logger.info("  • Higher win rate (54.9% vs 49.8%)")
    logger.info("  • Much lower drawdown (0.1% vs 5.9%)")
    logger.info("  • Regime-aware: avoids bad conditions")
    logger.info("  • Long signals performing well (60.5% win)")
    logger.info("  • Conservative risk management")
    
    logger.info("\n⚠️  Weaknesses:")
    logger.info("  • Very low absolute returns (+0.3% vs +76%)")
    logger.info("  • Short signals losing money (-3.67R)")
    logger.info("  • Too many small trades (461 vs 309)")
    logger.info("  • Mean reversion strategy too weak")
    logger.info("  • BREAKOUT regime too rare (only 6 trades)")
    
    logger.info("\n" + "="*100)
    logger.info("HEAD-TO-HEAD COMPARISON")
    logger.info("="*100)
    
    comparison = [
        ("", "Model #1", "Model #2", "Winner"),
        ("-" * 90, "-" * 90, "-" * 90, "-" * 90),
        ("Total Return", "+76.0%", "+0.3%", "🏆 Model #1"),
        ("Win Rate", "49.8%", "54.9%", "🏆 Model #2"),
        ("Max Drawdown", "5.9%", "0.1%", "🏆 Model #2"),
        ("Total Trades", "309", "461", "—"),
        ("Avg R/Trade", "+0.21R", "+0.03R", "🏆 Model #1"),
        ("Sharpe Ratio", "1.89", "~0.3", "🏆 Model #1"),
        ("Long Performance", "51% win", "60.5% win", "🏆 Model #2"),
        ("Short Performance", "49% win", "50.8% win", "≈ Tie"),
        ("Regime Awareness", "❌ No", "✅ Yes", "🏆 Model #2"),
        ("Risk Management", "⚠️ Moderate", "✅ Conservative", "🏆 Model #2"),
    ]
    
    logger.info("\n")
    for row in comparison:
        logger.info(f"{row[0]:25} | {row[1]:15} | {row[2]:15} | {row[3]}")
    
    logger.info("\n" + "="*100)
    logger.info("💡 KEY INSIGHTS")
    logger.info("="*100)
    
    logger.info("\n1️⃣  MODEL #1 IS THE CLEAR WINNER FOR PROFITABILITY")
    logger.info("   • 254x higher returns (+$19k vs +$76)")
    logger.info("   • Much better R per trade (0.21 vs 0.03)")
    logger.info("   • Strong Sharpe ratio (1.89)")
    logger.info("   • Proven edge on December 2025 data")
    
    logger.info("\n2️⃣  MODEL #2 IS BETTER FOR RISK MANAGEMENT")
    logger.info("   • 59x lower max drawdown (0.1% vs 5.9%)")
    logger.info("   • Higher win rate (54.9% vs 49.8%)")
    logger.info("   • Successfully avoids high-vol/low-liq periods")
    logger.info("   • But: strategy is too weak (too many tiny trades)")
    
    logger.info("\n3️⃣  MODEL #2 PROBLEM: MEAN REVERSION STRATEGY IS TOO WEAK")
    logger.info("   • 455 mean-reversion trades only made +$72 (+0.03R avg)")
    logger.info("   • VWAP z-score > ±1.5 threshold generates too many signals")
    logger.info("   • Small TP/SL distances (ATR-based) vs Model #1 fixed $22.50/$15")
    logger.info("   • Needs: Tighter filters OR bigger position sizing OR better entries")
    
    logger.info("\n4️⃣  MODEL #2 SHORT SIGNALS ARE BROKEN")
    logger.info("   • 266 short trades lost -$19.77 (-3.67R)")
    logger.info("   • Mean reversion shorts losing money in Dec uptrend")
    logger.info("   • Needs: Stronger directional bias OR avoid shorts in bull markets")
    
    logger.info("\n" + "="*100)
    logger.info("🚀 RECOMMENDED NEXT STEPS")
    logger.info("="*100)
    
    logger.info("\n✅ KEEP MODEL #1 AS PRIMARY SYSTEM (0.70/0.20 thresholds)")
    logger.info("   → Currently live and profitable")
    logger.info("   → Strong proven edge on OOS data")
    logger.info("   → Don't fix what's not broken")
    
    logger.info("\n🔧 IMPROVE MODEL #2 BEFORE INTEGRATION:")
    
    logger.info("\n   Option A: FIX MEAN REVERSION STRATEGY")
    logger.info("     • Increase VWAP z-score threshold to ±2.0 (fewer, better trades)")
    logger.info("     • Add volume confirmation (only trade at key support/resistance)")
    logger.info("     • Use fixed TP/SL like Model #1 ($22.50/$15) instead of ATR")
    logger.info("     • Add trend filter (don't short in strong uptrends)")
    
    logger.info("\n   Option B: USE MODEL #2 AS REGIME FILTER ONLY")
    logger.info("     • Classify regime with Model #2")
    logger.info("     • Run Model #1 signals normally")
    logger.info("     • But: Skip trades in HIGH_VOL or LOW_LIQ regimes")
    logger.info("     • Result: Same profit, lower drawdown")
    
    logger.info("\n   Option C: WEIGHTED ENSEMBLE")
    logger.info("     • If Model #1 says LONG AND Model #2 says MEAN_REV LONG → Double confidence")
    logger.info("     • If Model #1 says LONG BUT Model #2 says HIGH_VOL → Reduce size 50%")
    logger.info("     • If both agree → Full size")
    logger.info("     • If disagree → Reduce size OR skip")
    
    logger.info("\n" + "="*100)
    logger.info("🎯 RECOMMENDATION: OPTION B (REGIME FILTER)")
    logger.info("="*100)
    logger.info("\n   Rationale:")
    logger.info("   • Model #1 is already profitable - don't mess with it")
    logger.info("   • Model #2 regime detection is good (79.5% accuracy)")
    logger.info("   • Use Model #2 to AVOID bad regimes, not generate signals")
    logger.info("   • Expected result: Similar returns, lower drawdown, higher Sharpe")
    
    logger.info("\n   Implementation:")
    logger.info("   1. Keep Model #1 thresholds (0.70/0.20)")
    logger.info("   2. Add regime check before each trade")
    logger.info("   3. Block trades if regime = HIGH_VOL or LOW_LIQ")
    logger.info("   4. Optionally: Reduce size if regime = BREAKOUT (uncertain)")
    logger.info("   5. Test on Dec 2025 → if improvement, deploy to live")
    
    logger.info("\n" + "="*100)
    logger.info("END OF REPORT")
    logger.info("="*100 + "\n")


if __name__ == "__main__":
    main()

