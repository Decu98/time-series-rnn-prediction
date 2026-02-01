"""
Moduł treningowy.

Zawiera funkcje straty i narzędzia pomocnicze dla procesu uczenia.
"""

from .losses import gaussian_nll_loss, multistep_gaussian_nll_loss

__all__ = [
    "gaussian_nll_loss",
    "multistep_gaussian_nll_loss",
]
