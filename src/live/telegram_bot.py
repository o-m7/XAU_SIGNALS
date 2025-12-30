#!/usr/bin/env python3
"""
Telegram Bot for Signal Notifications.

Sends formatted trading signals to a Telegram chat.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

logger = logging.getLogger("TelegramBot")


class TelegramBot:
    """
    Telegram notification bot for trading signals.
    
    Args:
        token: Telegram Bot API token
        chat_id: Target chat/channel ID
        enabled: Whether to actually send messages (default True)
    """
    
    API_URL = "https://api.telegram.org/bot{token}/sendMessage"
    
    # Signal emoji mapping
    SIGNAL_EMOJI = {
        "LONG": "🟢",
        "SHORT": "🔴",
        "FLAT": "⚪",
    }
    
    def __init__(
        self,
        token: str,
        chat_id: str,
        enabled: bool = True
    ):
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled
        
        if not token or not chat_id:
            logger.warning("Telegram bot not fully configured")
            self.enabled = False
        else:
            logger.info(f"TelegramBot initialized for chat: {chat_id}")
    
    def send_signal(
        self,
        signal: str,
        proba_up: float,
        timestamp: Optional[datetime] = None,
        price: Optional[float] = None,
        tp: Optional[float] = None,
        sl: Optional[float] = None,
        risk_pct: float = 0.0025,
        model_name: str = "y_tb_60",
        extra_info: Optional[Dict] = None
    ) -> bool:
        """
        Send a trading signal to Telegram.

        Args:
            signal: Signal type (LONG, SHORT, FLAT)
            proba_up: Probability of up move
            timestamp: Signal timestamp
            price: Current price
            tp: Take Profit price
            sl: Stop Loss price
            risk_pct: Risk per trade
            model_name: Model identifier
            extra_info: Additional info to include

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.warning(f"Telegram disabled - signal {signal} NOT sent (token={bool(self.token)}, chat_id={bool(self.chat_id)})")
            return False
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Format message
        emoji = self.SIGNAL_EMOJI.get(signal, "❓")
        
        # Confidence display
        if signal == "LONG":
            confidence = proba_up
        elif signal == "SHORT":
            confidence = 1 - proba_up
        else:
            confidence = abs(0.5 - proba_up) * 2  # Distance from 0.5
        
        # Build message
        lines = [
            f"{emoji} <b>XAUUSD LIVE SIGNAL</b> {emoji}",
            "",
            f"<b>Signal:</b> {signal}",
            f"<b>Confidence:</b> {confidence:.2f}",
        ]
        
        if price:
            lines.append(f"<b>Entry:</b> {price:.2f}")
        
        # Add TP and SL with pip distance
        if tp is not None and sl is not None and price:
            if signal == "LONG":
                tp_pips = (tp - price) * 10  # Gold is quoted in dollars, ~10 pips per dollar
                sl_pips = (price - sl) * 10
            else:  # SHORT
                tp_pips = (price - tp) * 10
                sl_pips = (sl - price) * 10
            
            lines.append(f"<b>TP:</b> {tp:.2f} (+{tp_pips:.0f} pips)")
            lines.append(f"<b>SL:</b> {sl:.2f} (-{sl_pips:.0f} pips)")
            
            # Risk:Reward ratio
            if sl_pips > 0:
                rr = tp_pips / sl_pips
                lines.append(f"<b>R:R:</b> 1:{rr:.1f}")
        
        # Model display name
        if "Model #1" in model_name or "Triple-Barrier" in model_name:
            model_display = "Model #1 (Triple-Barrier)"
        elif "Model #3" in model_name or "CMF" in model_name or "MACD" in model_name:
            model_display = "Model #3 (CMF/MACD)"
        else:
            model_display = model_name
        
        lines.extend([
            "",
            f"<b>Model:</b> {model_display}",
            f"<b>Time:</b> {timestamp.strftime('%H:%M UTC')}",
            f"<b>Risk:</b> {risk_pct*100:.2f}%",
            f"<b>Account Mode:</b> Funded",
        ])
        
        if extra_info:
            lines.append("")
            for key, value in extra_info.items():
                lines.append(f"<b>{key}:</b> {value}")
        
        message = "\n".join(lines)
        
        return self._send_message(message)
    
    def send_alert(self, title: str, message: str) -> bool:
        """
        Send a generic alert message.
        
        Args:
            title: Alert title
            message: Alert body
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        text = f"⚠️ <b>{title}</b>\n\n{message}"
        return self._send_message(text)
    
    def send_status(self, status_dict: Dict) -> bool:
        """
        Send a status update message.
        
        Args:
            status_dict: Dict with status info
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        lines = ["📊 <b>Status Update</b>", ""]
        
        for key, value in status_dict.items():
            if isinstance(value, float):
                lines.append(f"<b>{key}:</b> {value:.4f}")
            else:
                lines.append(f"<b>{key}:</b> {value}")
        
        message = "\n".join(lines)
        return self._send_message(message)
    
    def _send_message(self, text: str) -> bool:
        """Send a message via Telegram API."""
        url = self.API_URL.format(token=self.token)
        
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_notification": False,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if result.get("ok"):
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {result}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test Telegram connection with a simple message."""
        return self.send_alert(
            "Connection Test",
            "XAUUSD Signal Engine connected successfully! 🚀"
        )


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    import os
    
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        exit(1)
    
    bot = TelegramBot(token=token, chat_id=chat_id)
    
    # Test connection
    print("Testing Telegram connection...")
    if bot.test_connection():
        print("✓ Connection successful")
    else:
        print("✗ Connection failed")
    
    # Test signal
    print("\nSending test signal...")
    bot.send_signal(
        signal="LONG",
        proba_up=0.72,
        price=2650.50,
        risk_pct=0.0025,
    )

