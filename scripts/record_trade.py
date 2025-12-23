#!/usr/bin/env python3
"""
Record Trade Outcomes for Reinforcement Learning.

This script records trade outcomes to improve future model training.
Call this after each signal's trade closes (TP hit, SL hit, or timeout).

Usage:
    python scripts/record_trade.py \\
        --timestamp "2024-01-15T10:30:00Z" \\
        --signal LONG \\
        --entry 2050.50 \\
        --tp 2070.50 \\
        --sl 2030.50 \\
        --outcome TP \\
        --exit 2070.50

Or use as a library:
    from src.retrain_weekly import record_trade_outcome
    record_trade_outcome(...)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrain_weekly import record_trade_outcome


def main():
    parser = argparse.ArgumentParser(
        description="Record a trade outcome for reinforcement learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--timestamp", required=True, help="Trade entry timestamp (ISO format)")
    parser.add_argument("--signal", required=True, choices=["LONG", "SHORT"], help="Trade direction")
    parser.add_argument("--entry", type=float, required=True, help="Entry price")
    parser.add_argument("--tp", type=float, required=True, help="Take profit price")
    parser.add_argument("--sl", type=float, required=True, help="Stop loss price")
    parser.add_argument("--outcome", required=True, choices=["TP", "SL", "TIMEOUT"], help="Trade outcome")
    parser.add_argument("--exit", type=float, required=True, help="Exit price")
    parser.add_argument("--duration", type=int, default=0, help="Duration in minutes")
    
    args = parser.parse_args()
    
    # Calculate R-multiple
    if args.signal == "LONG":
        risk = args.entry - args.sl
        pnl = args.exit - args.entry
    else:  # SHORT
        risk = args.sl - args.entry
        pnl = args.entry - args.exit
    
    r_multiple = pnl / risk if risk > 0 else 0
    
    record_trade_outcome(
        timestamp=args.timestamp,
        signal=args.signal,
        entry_price=args.entry,
        tp=args.tp,
        sl=args.sl,
        outcome=args.outcome,
        exit_price=args.exit,
        r_multiple=r_multiple,
        duration_minutes=args.duration
    )
    
    print(f"✅ Recorded: {args.signal} -> {args.outcome} ({r_multiple:+.2f}R)")


if __name__ == "__main__":
    main()

