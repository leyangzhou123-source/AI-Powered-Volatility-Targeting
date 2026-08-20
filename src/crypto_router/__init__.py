"""Independent crypto volatility routing package.

This package contains the BTC/USDT-specific estimators, controllers, and AI
router used by the crypto volatility-routing workflows.
"""

from src.crypto_router.ai_vol_target_router import (
    BTCBenchmarkGuardRouter,
    BTCVolTargetAIRouter,
    CryptoVolTargetAIRouter,
)
from src.crypto_router.controllers import PegAwareVolController, VolatilityShockThrottle
from src.crypto_router.estimators import (
    CryptoCompositeVolEstimator,
    IntradayRealizedVolEstimator,
    RangeBasedVolEstimator,
)

__all__ = [
    "BTCBenchmarkGuardRouter",
    "BTCVolTargetAIRouter",
    "CryptoCompositeVolEstimator",
    "CryptoVolTargetAIRouter",
    "IntradayRealizedVolEstimator",
    "PegAwareVolController",
    "RangeBasedVolEstimator",
    "VolatilityShockThrottle",
]
