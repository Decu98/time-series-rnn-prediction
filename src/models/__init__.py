"""
Moduł architektury modelu.

Zawiera implementacje:
- Encoder LSTM (z opcjonalnym conditioningiem)
- Decoder z wyjściem Gaussowskim
- Model Seq2Seq jako LightningModule
- Model Seq2Seq z parametrami warunkującymi (dla parametryzacji bezwymiarowej)
"""

from .decoder import GaussianDecoder
from .encoder import LSTMEncoder
from .seq2seq import Seq2SeqModel, ConditionedSeq2SeqModel

__all__ = [
    "LSTMEncoder",
    "GaussianDecoder",
    "Seq2SeqModel",
    "ConditionedSeq2SeqModel",
]
