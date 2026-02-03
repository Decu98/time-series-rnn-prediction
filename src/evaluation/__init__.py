"""
Moduł ewaluacji i wizualizacji.

Zawiera metryki oceny jakości predykcji oraz funkcje
do wizualizacji wyników.
"""

from .metrics import compute_coverage_probability, compute_mae, compute_nll, compute_rmse
from .visualization import (
    plot_phase_space,
    plot_prediction_with_uncertainty,
    plot_training_curves,
    plot_multi_prediction_trajectory,
)

__all__ = [
    "compute_rmse",
    "compute_mae",
    "compute_nll",
    "compute_coverage_probability",
    "plot_prediction_with_uncertainty",
    "plot_training_curves",
    "plot_phase_space",
    "plot_multi_prediction_trajectory",
]
