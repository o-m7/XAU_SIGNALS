"""
Backtest vs Live Performance Analysis

Investigates why Model #1 shows +76% in backtest but isn't profitable live.

Possible causes:
1. Lookahead bias in backtest
2. Slippage/spread not modeled correctly
3. Overfitting to December 2025
4. Execution timing issues
5. TP/SL hit detection errors
6. Different feature calculations (1-min live vs 5-min backtest)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def check_backtest_methodology():
    """Audit the backtest for potential issues."""
    logger.info("="*100)
    logger.info("BACKTEST AUDIT: POTENTIAL ISSUES WITH MODEL #1")
    logger.info("="*100)
    
    logger.info("\n🚨 USER REPORT: Model #1 NOT profitable live trading")
    logger.info("   Backtest: +76% (December 2025)")
    logger.info("   Live:     Not profitable")
    logger.info("   → Major red flag: backtest-to-live disconnect")
    
    logger.info("\n" + "="*100)
    logger.info("POTENTIAL CAUSES")
    logger.info("="*100)
    
    logger.info("\n1️⃣  LOOKAHEAD BIAS (Most likely)")
    logger.info("   Problem: Model might be seeing future data during prediction")
    logger.info("   ")
    logger.info("   Suspects:")
    logger.info("   a) Triple-barrier labels (y_tb_60) look 60 bars AHEAD")
    logger.info("      - Label created using future price movements")
    logger.info("      - At prediction time, we don't know which barrier will hit")
    logger.info("      - Backtest assumes we KNOW the outcome (which barrier hit first)")
    logger.info("   ")
    logger.info("   b) Features might include forward-looking data")
    logger.info("      - Rolling calculations using future bars")
    logger.info("      - Indicators with future data leakage")
    logger.info("   ")
    logger.info("   c) Backtest exit logic might be optimistic")
    logger.info("      - Assumes TP/SL hit at exact levels")
    logger.info("      - Doesn't model slippage or spread")
    logger.info("      - May allow entry on close, exit on same bar")
    
    logger.info("\n2️⃣  SLIPPAGE & EXECUTION COSTS")
    logger.info("   Problem: Backtest doesn't model real-world execution")
    logger.info("   ")
    logger.info("   Missing costs:")
    logger.info("   • Bid-ask spread: ~$0.50-2.00 per trade (entry + exit)")
    logger.info("   • Slippage: Fast-moving market, can't get exact TP/SL")
    logger.info("   • Latency: Signal generated → order placed → filled (1-5 seconds)")
    logger.info("   • Requotes: Price moved before order filled")
    logger.info("   ")
    logger.info("   Impact on 309 trades:")
    logger.info("   • If avg $1 slippage per trade → -$309 → -1.2% return")
    logger.info("   • If avg $2 slippage per trade → -$618 → -2.5% return")
    logger.info("   • Total: Could reduce +76% to +73.5% (still profitable)")
    logger.info("   • BUT: Not enough to explain unprofitable live trading")
    
    logger.info("\n3️⃣  OVERFITTING TO DECEMBER 2025")
    logger.info("   Problem: Model trained on 2024-2025, tested on Dec 2025")
    logger.info("   ")
    logger.info("   Concerns:")
    logger.info("   • Dec 2025 might have been an unusually good month")
    logger.info("   • Market conditions: Trending? Volatile? Range-bound?")
    logger.info("   • Model might exploit specific Dec patterns that don't repeat")
    logger.info("   ")
    logger.info("   Test: Check live trading month")
    logger.info("   • If live trading in Jan 2026 → different conditions")
    logger.info("   • If market regime changed → model edge disappears")
    
    logger.info("\n4️⃣  FEATURE CALCULATION MISMATCH")
    logger.info("   Problem: Live features ≠ Backtest features")
    logger.info("   ")
    logger.info("   Known issues:")
    logger.info("   • Backtest: Uses complete 1-minute bars, clean data")
    logger.info("   • Live: Real-time tick aggregation, partial bars")
    logger.info("   • Live: FutureWarnings suggest pandas compatibility issues")
    logger.info("   • Live: Rolling window calculations on growing buffer")
    logger.info("   ")
    logger.info("   If features drift even 5%, model predictions become unreliable")
    
    logger.info("\n5️⃣  TP/SL HIT DETECTION")
    logger.info("   Problem: Backtest optimistically assumes TP hit")
    logger.info("   ")
    logger.info("   Backtest logic (scripts/retrain_all_data.py):")
    logger.info("   • Looks at future 60 bars")
    logger.info("   • If high >= TP OR low <= SL → assumes hit")
    logger.info("   • Problem: What if both hit in same bar? Order matters!")
    logger.info("   • Backtest might favor TP when SL hit first in reality")
    logger.info("   ")
    logger.info("   Live reality:")
    logger.info("   • Price might spike through TP, then reverse to SL")
    logger.info("   • Or hit SL first, never reach TP")
    logger.info("   • Backtest can't model intra-bar price path")
    
    logger.info("\n6️⃣  SIGNAL TIMING ISSUES")
    logger.info("   Problem: Live signals generated on different bar close")
    logger.info("   ")
    logger.info("   Backtest:")
    logger.info("   • Signal on bar N close")
    logger.info("   • Enter at bar N close price")
    logger.info("   • Exit when TP/SL hit in future bars")
    logger.info("   ")
    logger.info("   Live:")
    logger.info("   • Signal on bar N close")
    logger.info("   • But: Next bar already started (price moved)")
    logger.info("   • Enter at bar N+1 open (worse price)")
    logger.info("   • Slippage: 1-bar delay = big difference")
    
    logger.info("\n7️⃣  COMMISSION & FEES")
    logger.info("   Problem: Backtest assumes zero cost")
    logger.info("   ")
    logger.info("   Real costs:")
    logger.info("   • Broker commission: $0-10 per trade (depends on broker)")
    logger.info("   • Swap/overnight fees: $5-20 per day if held overnight")
    logger.info("   • Platform fees: Monthly or per-trade")
    logger.info("   ")
    logger.info("   309 trades × $5 avg = $1,545 in fees")
    logger.info("   This would reduce +76% significantly")
    
    logger.info("\n" + "="*100)
    logger.info("CRITICAL BACKTEST FLAWS FOUND")
    logger.info("="*100)
    
    logger.info("\n🔴 FLAW #1: TRIPLE-BARRIER LABELS ARE INHERENTLY FORWARD-LOOKING")
    logger.info("   The y_tb_60 label is created by looking 60 bars AHEAD and checking")
    logger.info("   which barrier (TP or SL) hit first. This is PERFECT INFORMATION.")
    logger.info("   ")
    logger.info("   In live trading:")
    logger.info("   • We predict P(label=+1) where +1 means 'TP will hit first'")
    logger.info("   • But we DON'T KNOW if TP will actually hit first")
    logger.info("   • We're predicting a label that depends on future outcomes")
    logger.info("   ")
    logger.info("   This is SUBTLE LOOKAHEAD BIAS:")
    logger.info("   • Model learns: 'When X features occur, TP usually hits first'")
    logger.info("   • But: Features might be correlated with DIRECTION, not TP/SL outcome")
    logger.info("   • Model might just be learning trend direction, not actual win/loss")
    
    logger.info("\n🔴 FLAW #2: BACKTEST USES SAME DATA AS TRAINING")
    logger.info("   Training: Jan 2024 - Nov 2025")
    logger.info("   Testing:  Dec 2025")
    logger.info("   ")
    logger.info("   Problems:")
    logger.info("   • Only 1 month out-of-sample (Dec 2025)")
    logger.info("   • If Dec 2025 had similar regime to training data → good results")
    logger.info("   • If live trading in different regime → poor results")
    logger.info("   • Need to test on MULTIPLE out-of-sample periods")
    
    logger.info("\n🔴 FLAW #3: BACKTEST DOESN'T MODEL REAL EXECUTION")
    logger.info("   Backtest assumes:")
    logger.info("   • Entry at exact close price")
    logger.info("   • TP/SL hit at exact levels")
    logger.info("   • No slippage, no spread, no commissions")
    logger.info("   • Instant execution")
    logger.info("   ")
    logger.info("   Reality:")
    logger.info("   • Entry 1-2 bars later (signal lag)")
    logger.info("   • Slippage: $0.50-2.00 per side")
    logger.info("   • Spread: $0.50-1.50 (bid-ask)")
    logger.info("   • Requotes: Price moved before fill")
    
    logger.info("\n" + "="*100)
    logger.info("RECOMMENDED ACTIONS")
    logger.info("="*100)
    
    logger.info("\n✅ ACTION 1: ANALYZE LIVE TRADING DATA")
    logger.info("   Collect:")
    logger.info("   • All live signals generated (date, time, direction, P(up))")
    logger.info("   • Actual entry prices")
    logger.info("   • Actual exit prices & reasons (TP/SL/timeout)")
    logger.info("   • Actual P&L per trade")
    logger.info("   ")
    logger.info("   Compare:")
    logger.info("   • Live win rate vs backtest win rate (49.8%)")
    logger.info("   • Live avg R vs backtest avg R (+0.21R)")
    logger.info("   • Live slippage per trade")
    logger.info("   ")
    logger.info("   Diagnose:")
    logger.info("   • If live win rate << 49.8% → model isn't working")
    logger.info("   • If live win rate ≈ 49.8% but P&L negative → execution costs too high")
    logger.info("   • If live signals rare → feature calculation mismatch")
    
    logger.info("\n✅ ACTION 2: FIX BACKTEST TO MODEL REALITY")
    logger.info("   Improvements:")
    logger.info("   • Add 1-bar entry delay (signal on bar N, enter bar N+1 open)")
    logger.info("   • Add slippage: $1-2 per trade (configurable)")
    logger.info("   • Add spread: $0.50-1.50 (configurable)")
    logger.info("   • Add commissions: $5-10 per trade")
    logger.info("   • Model intra-bar TP/SL ambiguity (random if both hit)")
    logger.info("   ")
    logger.info("   Re-run backtest with realistic costs:")
    logger.info("   • Expected: +76% drops to +20-40% (still profitable?)")
    logger.info("   • If drops to negative → model has no real edge")
    
    logger.info("\n✅ ACTION 3: TEST ON MULTIPLE OUT-OF-SAMPLE PERIODS")
    logger.info("   Don't just test on Dec 2025:")
    logger.info("   • Test on Jan-Nov 2024 (each month separately)")
    logger.info("   • Test on each quarter of 2025")
    logger.info("   • Check consistency across periods")
    logger.info("   ")
    logger.info("   If profitable only in Dec 2025:")
    logger.info("   • Model is overfit to that specific month")
    logger.info("   • Need more robust features or regularization")
    
    logger.info("\n✅ ACTION 4: COMPARE LIVE FEATURES VS BACKTEST FEATURES")
    logger.info("   Log live features to CSV:")
    logger.info("   • Every time signal generated, log all 43 features")
    logger.info("   • Compare to same timestamp in backtest data")
    logger.info("   • Check for mismatches (>1% difference = problem)")
    logger.info("   ")
    logger.info("   Common issues:")
    logger.info("   • Live: Rolling window incomplete (first N bars)")
    logger.info("   • Live: Timezone issues (UTC vs local)")
    logger.info("   • Live: Partial bar (using bar before close)")
    
    logger.info("\n✅ ACTION 5: SIMPLIFY STRATEGY")
    logger.info("   If Model #1 isn't working:")
    logger.info("   • Maybe it's too complex (43 features, 60-bar horizon)")
    logger.info("   • Try simpler approach:")
    logger.info("     - Shorter horizon (15-30 bars)")
    logger.info("     - Fewer features (10-15 most important)")
    logger.info("     - Wider thresholds (0.75/0.15 instead of 0.70/0.20)")
    logger.info("   ")
    logger.info("   Or: Use Model #2 regime filter more aggressively")
    logger.info("   • Only trade in MEAN_REVERTING regime")
    logger.info("   • Skip everything else")
    logger.info("   • Fewer trades, but higher quality")
    
    logger.info("\n" + "="*100)
    logger.info("IMMEDIATE NEXT STEPS")
    logger.info("="*100)
    
    logger.info("\n1. Check live_runner.log for recent signals")
    logger.info("2. Calculate live win rate from Telegram messages")
    logger.info("3. Measure actual slippage (entry price vs expected)")
    logger.info("4. Re-run backtest with realistic execution costs")
    logger.info("5. Compare live features vs backtest features for same timestamps")
    
    logger.info("\n" + "="*100)
    logger.info("CONCLUSION")
    logger.info("="*100)
    
    logger.info("\n⚠️  Model #1's +76% backtest result is likely MISLEADING")
    logger.info("   ")
    logger.info("   Reasons:")
    logger.info("   • Triple-barrier labels have subtle lookahead bias")
    logger.info("   • Backtest doesn't model real execution costs")
    logger.info("   • Only tested on 1 month (Dec 2025)")
    logger.info("   • Feature calculation might differ live vs backtest")
    logger.info("   ")
    logger.info("   Real-world profitability is likely much lower or negative.")
    logger.info("   ")
    logger.info("   We need to:")
    logger.info("   1. Fix the backtest methodology")
    logger.info("   2. Test on multiple periods")
    logger.info("   3. Analyze actual live performance data")
    logger.info("   4. Find the root cause of backtest-live disconnect")
    
    logger.info("\n")


def main():
    check_backtest_methodology()
    
    logger.info("="*100)
    logger.info("Would you like me to:")
    logger.info("1. Analyze your live trading logs to diagnose the issue?")
    logger.info("2. Fix the backtest to add realistic execution costs?")
    logger.info("3. Test Model #1 on multiple periods (not just Dec 2025)?")
    logger.info("4. Compare live feature calculations vs backtest?")
    logger.info("="*100 + "\n")


if __name__ == "__main__":
    main()

