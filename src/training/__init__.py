"""
Moduł treningowy.

Zawiera funkcje straty i narzędzia pomocnicze dla procesu uczenia.
Obsługuje zarówno PyTorch Lightning jak i ręczną pętlę dla DirectML.
"""

from .losses import gaussian_nll_loss, multistep_gaussian_nll_loss
from .manual_trainer import ManualTrainer, create_manual_trainer

__all__ = [
    "gaussian_nll_loss",
    "multistep_gaussian_nll_loss",
    "ManualTrainer",
    "create_manual_trainer",
]
