"""
Moduł konfiguracyjny projektu.

Zawiera dataclassy z hiperparametrami dla:
- generacji danych syntetycznych
- preprocessingu
- architektury modelu
- procesu treningowego
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DataConfig:
    """
    Konfiguracja generacji i przetwarzania danych.

    Attributes:
        dt: Krok czasowy symulacji [s]
        t_max: Maksymalny czas symulacji [s]
        num_trajectories: Liczba trajektorii do wygenerowania
        noise_std: Odchylenie standardowe szumu gaussowskiego
        train_ratio: Proporcja danych treningowych
        val_ratio: Proporcja danych walidacyjnych
        test_ratio: Proporcja danych testowych
    """
    dt: float = 0.01
    t_max: float = 10.0
    num_trajectories: int = 1000
    noise_std: float = 0.01
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class OscillatorConfig:
    """
    Konfiguracja parametrów oscylatora tłumionego.

    Równanie ruchu: m*x'' + c*x' + k*x = 0

    Attributes:
        mass_range: Zakres masy [kg]
        damping_range: Zakres współczynnika tłumienia [Ns/m]
        stiffness_range: Zakres sztywności [N/m]
        x0_range: Zakres początkowego położenia [m]
        v0_range: Zakres początkowej prędkości [m/s]
    """
    mass_range: Tuple[float, float] = (1.0, 1.0)
    damping_range: Tuple[float, float] = (0.1, 0.5)
    stiffness_range: Tuple[float, float] = (1.0, 5.0)
    x0_range: Tuple[float, float] = (-2.0, 2.0)
    v0_range: Tuple[float, float] = (-1.0, 1.0)


@dataclass
class PreprocessingConfig:
    """
    Konfiguracja preprocessingu danych.

    Attributes:
        apply_filter: Czy stosować filtrowanie sygnału
        filter_window: Rozmiar okna filtru Savitzky-Golay
        filter_order: Rząd wielomianu filtru
        normalize: Czy normalizować dane
    """
    apply_filter: bool = False
    filter_window: int = 5
    filter_order: int = 2
    normalize: bool = True


@dataclass
class ModelConfig:
    """
    Konfiguracja architektury modelu Seq2Seq.

    Attributes:
        input_size: Liczba cech wejściowych (x, v)
        hidden_size: Rozmiar warstwy ukrytej LSTM
        num_layers: Liczba warstw LSTM
        dropout: Współczynnik dropout (między warstwami LSTM)
        bidirectional_encoder: Czy encoder ma być dwukierunkowy
    """
    input_size: int = 2  # pozycja x, prędkość v
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional_encoder: bool = False


@dataclass
class TrainingConfig:
    """
    Konfiguracja procesu treningowego.

    Attributes:
        T_in: Długość okna wejściowego (liczba kroków czasowych)
        T_out: Długość horyzontu predykcji (liczba kroków czasowych)
        batch_size: Rozmiar batcha
        learning_rate: Współczynnik uczenia
        max_epochs: Maksymalna liczba epok
        teacher_forcing_ratio: Początkowy współczynnik teacher forcing
        teacher_forcing_decay: Współczynnik zaniku teacher forcing na epokę
        gradient_clip_val: Maksymalna norma gradientu
        early_stopping_patience: Cierpliwość dla early stopping
        min_sigma: Minimalna wartość sigma (stabilność numeryczna)
    """
    T_in: int = 50
    T_out: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 100
    teacher_forcing_ratio: float = 1.0
    teacher_forcing_decay: float = 0.02
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10
    min_sigma: float = 1e-4


@dataclass
class Config:
    """
    Główna klasa konfiguracyjna agregująca wszystkie ustawienia.

    Attributes:
        data: Konfiguracja danych
        oscillator: Konfiguracja oscylatora
        preprocessing: Konfiguracja preprocessingu
        model: Konfiguracja modelu
        training: Konfiguracja treningu
        seed: Ziarno generatora losowego dla reprodukowalności
        device: Urządzenie obliczeniowe ('cpu', 'cuda', 'mps')
    """
    data: DataConfig = field(default_factory=DataConfig)
    oscillator: OscillatorConfig = field(default_factory=OscillatorConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self):
        """Walidacja konfiguracji po inicjalizacji."""
        # Sprawdzenie proporcji podziału danych
        total_ratio = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Suma proporcji podziału danych musi wynosić 1.0, "
                f"otrzymano: {total_ratio}"
            )

        # Sprawdzenie parametrów modelu
        if self.model.hidden_size <= 0:
            raise ValueError("hidden_size musi być większy od 0")

        if self.model.num_layers <= 0:
            raise ValueError("num_layers musi być większy od 0")


def get_default_config() -> Config:
    """
    Zwraca domyślną konfigurację projektu.

    Returns:
        Config: Obiekt konfiguracji z domyślnymi wartościami
    """
    return Config()
