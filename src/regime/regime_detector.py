#!/usr/bin/env python3
"""
Regime Detection for XAUUSD Intraday Trading.

Classifies market into 3 regimes:
- TRENDING (directional, use trend-following)
- RANGING (mean-reverting, trade extremes)
- VOLATILE (choppy, skip or reduce size)
"""

import numpy as np
import pandas as pd
from typing import Tuple


class RegimeDetector:
    """
    Detects market regime using multiple indicators:
    - ADX (trend strength)
    - ATR percentile (volatility regime)
    - Efficiency ratio (trend vs noise)
    """
    
    def __init__(
        self,
        adx_window: int = 14,
        adx_trending_threshold: float = 25.0,
        adx_strong_threshold: float = 40.0,
        atr_window: int = 14,
        atr_lookback: int = 100,
        atr_high_pct: float = 0.75,
        efficiency_window: int = 20
    ):
        self.adx_window = adx_window
        self.adx_trending = adx_trending_threshold
        self.adx_strong = adx_strong_threshold
        self.atr_window = atr_window
        self.atr_lookback = atr_lookback
        self.atr_high_pct = atr_high_pct
        self.efficiency_window = efficiency_window
    
    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ADX (Average Directional Index)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed indicators
        atr = tr.ewm(span=self.adx_window, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=self.adx_window, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=self.adx_window, adjust=False).mean() / atr
        
        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
        adx = dx.ewm(span=self.adx_window, adjust=False).mean()
        
        return adx.fillna(0)
    
    def calculate_efficiency_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Efficiency Ratio (Kaufman).
        ER = Net Change / Sum of Absolute Changes
        
        High ER = Trending
        Low ER = Choppy
        """
        close = df['close']
        
        net_change = (close - close.shift(self.efficiency_window)).abs()
        sum_changes = close.diff().abs().rolling(self.efficiency_window).sum()
        
        er = net_change / (sum_changes + 1e-8)
        return er.fillna(0)
    
    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect regime for each bar.
        
        Returns DataFrame with columns:
        - regime: 'TRENDING', 'RANGING', 'VOLATILE'
        - regime_strength: 0.0 to 1.0
        - adx, atr_pct, efficiency_ratio (for debugging)
        """
        df = df.copy()
        
        # Calculate indicators
        df['adx'] = self.calculate_adx(df)
        df['efficiency_ratio'] = self.calculate_efficiency_ratio(df)
        
        # ATR percentile (current ATR vs historical distribution)
        if 'ATR_14' not in df.columns:
            # Calculate ATR if not present
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR_14'] = tr.rolling(self.atr_window).mean()
        
        # Calculate ATR percentile
        df['atr_pct'] = df['ATR_14'].rolling(self.atr_lookback, min_periods=10).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        )
        df['atr_pct'] = df['atr_pct'].fillna(0.5)
        
        # Regime classification
        regime = []
        strength = []
        
        for i in range(len(df)):
            adx_val = df['adx'].iloc[i] if not pd.isna(df['adx'].iloc[i]) else 0
            atr_pct_val = df['atr_pct'].iloc[i] if not pd.isna(df['atr_pct'].iloc[i]) else 0.5
            er_val = df['efficiency_ratio'].iloc[i] if not pd.isna(df['efficiency_ratio'].iloc[i]) else 0
            
            # VOLATILE: High ATR percentile (top 25%)
            if atr_pct_val > self.atr_high_pct:
                regime.append('VOLATILE')
                strength.append(min(atr_pct_val, 1.0))
            
            # TRENDING: Strong ADX + High Efficiency
            elif adx_val > self.adx_strong:
                regime.append('TRENDING')
                strength.append(min(adx_val / 50.0, 1.0))
            
            # WEAK TRENDING: Moderate ADX
            elif adx_val > self.adx_trending and er_val > 0.3:
                regime.append('TRENDING')
                strength.append(min(adx_val / 50.0, 1.0))
            
            # RANGING: Low ADX, Low Efficiency
            else:
                regime.append('RANGING')
                strength.append(max(0.0, 1.0 - (adx_val / self.adx_trending)))
        
        df['regime'] = regime
        df['regime_strength'] = strength
        
        return df
    
    def should_trade(self, regime: str, session: str = None) -> bool:
        """
        Determine if we should trade given regime and session.
        
        Rules:
        - TRENDING: Trade all sessions (best)
        - RANGING: Trade during London/NY open (liquidity)
        - VOLATILE: Skip (or reduce size 50%)
        """
        if regime == 'VOLATILE':
            return False  # Skip volatile periods
        
        if regime == 'RANGING':
            # Only trade ranging during high liquidity sessions
            if session in ['is_europe', 'is_us']:
                return True
            return False
        
        return True  # TRENDING: always trade

