"""
Moduł obsługi danych dla PyTorch.

Zawiera implementacje Dataset i DataModule dla szeregów czasowych.
"""

from .time_series_dataset import TimeSeriesDataModule, TimeSeriesDataset

__all__ = [
    "TimeSeriesDataset",
    "TimeSeriesDataModule",
]
