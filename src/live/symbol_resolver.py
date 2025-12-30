#!/usr/bin/env python3
"""
Centralized Symbol Resolver for Polygon.io.

Handles the different symbol formats required by Polygon's
WebSocket and REST APIs for forex pairs.

WebSocket symbols:
    - Quotes: C.XAU/USD
    - Minute aggregates: CA.XAU/USD
    - Second aggregates: CAS.XAU/USD

REST symbols:
    - Snapshot: C:XAUUSD
    - Aggregates: C:XAUUSD
    - Last quote: ("XAU", "USD") as separate args
    - Quotes list: C:XAU-USD
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SymbolResolver:
    """
    Centralized symbol resolver for Polygon forex symbols.
    
    Handles the different symbol formats required by Polygon's
    WebSocket and REST APIs.
    
    Args:
        base: Base currency (e.g., "XAU" for gold)
        quote: Quote currency (e.g., "USD")
    
    Example:
        resolver = SymbolResolver("XAU", "USD")
        ws_channel = resolver.ws_quotes()  # "C.XAU/USD"
        rest_sym = resolver.rest_aggs_symbol()  # "C:XAUUSD"
    """
    base: str = "XAU"
    quote: str = "USD"
    
    def ws_quotes(self) -> str:
        """WebSocket channel for forex quotes: C:XAU-USD"""
        return f"C:{self.base}-{self.quote}"
    
    def ws_aggs_minute(self) -> str:
        """WebSocket channel for minute aggregates: CA.XAU/USD"""
        return f"CA.{self.base}/{self.quote}"
    
    def ws_aggs_second(self) -> str:
        """WebSocket channel for second aggregates: CAS.XAU/USD"""
        return f"CAS.{self.base}/{self.quote}"
    
    def rest_snapshot(self) -> str:
        """REST symbol for snapshot endpoint: C:XAUUSD"""
        return f"C:{self.base}{self.quote}"
    
    def rest_quotes_symbol(self) -> str:
        """REST symbol for quotes list endpoint: C:XAU-USD"""
        return f"C:{self.base}-{self.quote}"
    
    def rest_last_quote_args(self) -> Tuple[str, str]:
        """
        Arguments for client.get_last_forex_quote().
        
        Returns:
            Tuple of (base, quote) e.g., ("XAU", "USD")
        """
        return (self.base, self.quote)
    
    def rest_aggs_symbol(self) -> str:
        """REST symbol for aggregates endpoint: C:XAUUSD"""
        return f"C:{self.base}{self.quote}"
    
    def display_name(self) -> str:
        """Human-readable display name: XAUUSD"""
        return f"{self.base}{self.quote}"
    
    def __str__(self) -> str:
        return self.display_name()


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    resolver = SymbolResolver("XAU", "USD")
    
    print("Symbol Resolver Test")
    print("=" * 40)
    print(f"Display Name:        {resolver.display_name()}")
    print()
    print("WebSocket Channels:")
    print(f"  Quotes:            {resolver.ws_quotes()}")
    print(f"  Minute Aggs:       {resolver.ws_aggs_minute()}")
    print(f"  Second Aggs:       {resolver.ws_aggs_second()}")
    print()
    print("REST Symbols:")
    print(f"  Snapshot:          {resolver.rest_snapshot()}")
    print(f"  Quotes List:       {resolver.rest_quotes_symbol()}")
    print(f"  Aggregates:        {resolver.rest_aggs_symbol()}")
    print(f"  Last Quote Args:   {resolver.rest_last_quote_args()}")

