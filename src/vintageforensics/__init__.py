"""Self-contained port of the retailPRED MRTS point-in-time evaluation stack.

Ported from retailPRED at revision f6a769adbfc65faca86aa34cb6160ad4e769427e
(the revision recorded in the failed 2026-11-04 vintage gate). This satellite
never imports retailPRED; behavioral equivalence is verified fold-for-fold
against the frozen gate artifacts in ``reference/``.
"""

from vintageforensics.harness import VintageBacktestResult, availability_as_of, vintage_panel_backtest
from vintageforensics.metrics import bias, mae, wape

__all__ = [
    "VintageBacktestResult",
    "availability_as_of",
    "vintage_panel_backtest",
    "bias",
    "mae",
    "wape",
]
