"""
Moduł obsługi danych dla PyTorch.

Zawiera implementacje Dataset i DataModule dla szeregów czasowych,
w tym wersje z parametrami warunkującymi dla parametryzacji bezwymiarowej.
"""

from .time_series_dataset import (
    TimeSeriesDataset,
    TimeSeriesDataModule,
    ConditionedTimeSeriesDataset,
    ConditionedTimeSeriesDataModule,
)

__all__ = [
    "TimeSeriesDataset",
    "TimeSeriesDataModule",
    "ConditionedTimeSeriesDataset",
    "ConditionedTimeSeriesDataModule",
]
