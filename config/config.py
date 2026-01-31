"""
Moduł konfiguracyjny projektu.

Zawiera dataclassy z hiperparametrami dla:
- generacji danych syntetycznych
- preprocessingu
- architektury modelu
- procesu treningowego

Obsługiwane platformy GPU:
- NVIDIA (CUDA)
- AMD (ROCm na Linux, DirectML na Windows)
- Apple Silicon (MPS)
"""

import platform
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


def is_directml_available() -> bool:
    """
    Sprawdza czy DirectML (AMD/Intel na Windows) jest dostępne.

    Returns:
        bool: True jeśli torch-directml jest zainstalowane i działa
    """
    try:
        import torch_directml
        return torch_directml.is_available()
    except ImportError:
        return False


def get_directml_device() -> Optional["torch.device"]:
    """
    Zwraca urządzenie DirectML jeśli dostępne.

    Returns:
        torch.device lub None
    """
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return None


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


def get_device(preferred: str = "auto") -> torch.device:
    """
    Wykrywa i zwraca optymalne urządzenie obliczeniowe.

    Obsługuje:
    - NVIDIA GPU (CUDA)
    - AMD GPU (ROCm na Linux, DirectML na Windows)
    - Apple Silicon (MPS)
    - CPU jako fallback

    Args:
        preferred: Preferowane urządzenie ('auto', 'cpu', 'cuda', 'mps', 'directml')

    Returns:
        torch.device: Wykryte urządzenie
    """
    if preferred == "cpu":
        return torch.device("cpu")

    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("UWAGA: CUDA niedostępne, używam CPU")
        return torch.device("cpu")

    if preferred == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                return torch.device("mps")
        print("UWAGA: MPS niedostępne, używam CPU")
        return torch.device("cpu")

    if preferred == "directml":
        dml_device = get_directml_device()
        if dml_device is not None:
            return dml_device
        print("UWAGA: DirectML niedostępne, używam CPU")
        return torch.device("cpu")

    # Auto-detekcja (preferred == "auto")
    # Priorytet: CUDA (NVIDIA/AMD ROCm) > DirectML (Windows) > MPS (Apple) > CPU

    # 1. CUDA (NVIDIA lub AMD ROCm na Linux)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Wykryto GPU (CUDA): {device_name}")
        return torch.device("cuda")

    # 2. DirectML (AMD/Intel na Windows)
    if platform.system() == "Windows":
        dml_device = get_directml_device()
        if dml_device is not None:
            print("Wykryto GPU (DirectML)")
            return dml_device

    # 3. MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("Wykryto Apple Silicon (MPS)")
            return torch.device("mps")

    print("Brak GPU, używam CPU")
    return torch.device("cpu")


def get_accelerator_config(preferred: str = "auto") -> dict:
    """
    Zwraca konfigurację akceleratora dla PyTorch Lightning.

    UWAGA: DirectML nie jest wspierany przez PyTorch Lightning.
    Dla DirectML używaj ręcznej pętli treningowej lub CPU w Lightning.

    Args:
        preferred: Preferowane urządzenie ('auto', 'cpu', 'cuda', 'mps', 'directml')

    Returns:
        dict: Słownik z kluczami 'accelerator' i 'devices' dla pl.Trainer
    """
    if preferred == "cpu":
        return {"accelerator": "cpu", "devices": 1}

    if preferred == "cuda":
        if torch.cuda.is_available():
            return {"accelerator": "gpu", "devices": 1}
        return {"accelerator": "cpu", "devices": 1}

    if preferred == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {"accelerator": "mps", "devices": 1}
        return {"accelerator": "cpu", "devices": 1}

    if preferred == "directml":
        # DirectML nie jest wspierany przez PyTorch Lightning
        # Użyj ręcznej pętli treningowej lub CPU
        print("UWAGA: DirectML nie jest wspierany przez PyTorch Lightning.")
        print("       Dla treningu na AMD GPU (Windows) użyj ręcznej pętli.")
        return {"accelerator": "cpu", "devices": 1}

    # Auto-detekcja
    if torch.cuda.is_available():
        return {"accelerator": "gpu", "devices": 1}

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return {"accelerator": "mps", "devices": 1}

    # DirectML w auto-detekcji - fallback do CPU dla Lightning
    if platform.system() == "Windows" and is_directml_available():
        print("UWAGA: DirectML wykryty, ale PyTorch Lightning go nie wspiera.")
        print("       Używam CPU dla Lightning. Rozważ ręczną pętlę treningową.")

    return {"accelerator": "cpu", "devices": 1}


def print_device_info() -> None:
    """
    Wyświetla informacje o dostępnych urządzeniach obliczeniowych.
    """
    print("\n" + "=" * 50)
    print("INFORMACJE O URZĄDZENIACH OBLICZENIOWYCH")
    print("=" * 50)

    print(f"\nSystem: {platform.system()} {platform.release()}")
    print(f"PyTorch wersja: {torch.__version__}")

    # CUDA (NVIDIA / AMD ROCm na Linux)
    print(f"\nCUDA dostępne: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda if torch.version.cuda else "ROCm"
        print(f"  - Wersja: {cuda_version}")
        print(f"  - Liczba GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  - GPU {i}: {props.name} ({memory_gb:.1f} GB)")

    # DirectML (AMD/Intel na Windows)
    dml_available = is_directml_available()
    print(f"\nDirectML (Windows AMD/Intel): {dml_available}")
    if dml_available:
        try:
            import torch_directml
            print(f"  - torch-directml zainstalowany")
            print(f"  - UWAGA: Wymaga ręcznej pętli treningowej (brak wsparcia Lightning)")
        except ImportError:
            pass

    # MPS (Apple Silicon)
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    print(f"\nMPS (Apple Silicon): {mps_available}")
    if mps_available:
        print(f"  - MPS zbudowane: {mps_built}")

    # Rekomendacja
    device = get_device("auto")
    print(f"\nRekomendowane urządzenie: {device}")
    print("=" * 50 + "\n")
