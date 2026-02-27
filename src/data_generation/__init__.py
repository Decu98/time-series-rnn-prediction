"""
Moduł generacji danych syntetycznych.

Zawiera implementacje generatorów danych dla układów dynamicznych,
w szczególności oscylatora harmonicznego tłumionego oraz
oscylatora w parametryzacji bezwymiarowej.
"""

from .synthetic import (
    DampedOscillator,
    SimpleHarmonicOscillator,
    DimensionlessOscillator,
    DimensionlessParams,
    ForcedDimensionlessOscillator,
    ForcedDimensionlessParams,
    add_noise,
    generate_dataset,
    generate_dimensionless_dataset,
    generate_forced_dimensionless_dataset,
    save_dataset,
    load_dataset,
    save_dimensionless_dataset,
    load_dimensionless_dataset,
)

__all__ = [
    "DampedOscillator",
    "SimpleHarmonicOscillator",
    "DimensionlessOscillator",
    "DimensionlessParams",
    "ForcedDimensionlessOscillator",
    "ForcedDimensionlessParams",
    "generate_dataset",
    "generate_dimensionless_dataset",
    "generate_forced_dimensionless_dataset",
    "add_noise",
    "save_dataset",
    "load_dataset",
    "save_dimensionless_dataset",
    "load_dimensionless_dataset",
]
