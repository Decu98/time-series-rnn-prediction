"""
Moduł konfiguracyjny projektu.
"""

from .config import (
    Config,
    DataConfig,
    ModelConfig,
    OscillatorConfig,
    PreprocessingConfig,
    TrainingConfig,
    get_default_config,
)

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "OscillatorConfig",
    "PreprocessingConfig",
    "TrainingConfig",
    "get_default_config",
]
