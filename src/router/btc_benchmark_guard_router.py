"""Compatibility imports for the crypto AI volatility router."""

from src.crypto_router.ai_vol_target_router import (
    BTCBenchmarkGuardRouter,
    BTCVolTargetAIRouter,
    CryptoVolTargetAIRouter,
)

__all__ = ["CryptoVolTargetAIRouter", "BTCVolTargetAIRouter", "BTCBenchmarkGuardRouter"]
