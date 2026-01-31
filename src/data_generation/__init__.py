"""
Moduł generacji danych syntetycznych.

Zawiera implementacje generatorów danych dla układów dynamicznych,
w szczególności oscylatora harmonicznego tłumionego.
"""

from .synthetic import DampedOscillator, add_noise, generate_dataset

__all__ = [
    "DampedOscillator",
    "generate_dataset",
    "add_noise",
]
