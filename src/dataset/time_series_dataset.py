"""
Moduł obsługi danych dla PyTorch.

Implementuje:
- TimeSeriesDataset: Dataset tworzący okna czasowe
- TimeSeriesDataModule: LightningDataModule do zarządzania danymi
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """
    Dataset PyTorch dla szeregów czasowych.

    Tworzy sliding windows z trajektorii, gdzie każda próbka
    składa się z okna wejściowego (input) i okna docelowego (target).

    Attributes:
        data: Znormalizowane dane trajektorii
        T_in: Długość okna wejściowego
        T_out: Długość okna docelowego (horyzont predykcji)
        stride: Krok przesuwania okna
        windows: Lista indeksów (trajectory_idx, start_idx) dla każdego okna
    """

    def __init__(
        self,
        data: np.ndarray,
        T_in: int,
        T_out: int,
        stride: int = 1
    ):
        """
        Inicjalizacja datasetu.

        Args:
            data: Dane trajektorii, kształt (num_trajectories, num_steps, features)
            T_in: Długość okna wejściowego (historia)
            T_out: Długość okna docelowego (predykcja)
            stride: Krok przesuwania okna (domyślnie 1)
        """
        self.data = torch.FloatTensor(data)
        self.T_in = T_in
        self.T_out = T_out
        self.stride = stride

        # Wymiary danych
        self.num_trajectories = data.shape[0]
        self.num_steps = data.shape[1]
        self.num_features = data.shape[2]

        # Minimalna długość trajektorii potrzebna dla jednego okna
        self.min_length = T_in + T_out

        # Tworzenie indeksów okien
        self.windows = self._create_windows()

    def _create_windows(self) -> List[Tuple[int, int]]:
        """
        Tworzy listę indeksów okien dla wszystkich trajektorii.

        Dla każdej trajektorii generuje wszystkie możliwe okna
        z zadanym krokiem (stride).

        Returns:
            Lista tupli (trajectory_idx, start_idx)
        """
        windows = []

        for traj_idx in range(self.num_trajectories):
            # Maksymalny indeks startowy dla tego okna
            max_start = self.num_steps - self.min_length

            # Generowanie indeksów z krokiem stride
            for start_idx in range(0, max_start + 1, self.stride):
                windows.append((traj_idx, start_idx))

        return windows

    def __len__(self) -> int:
        """Zwraca całkowitą liczbę okien w datasecie."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pobiera pojedynczą próbkę (okno wejściowe i docelowe).

        Args:
            idx: Indeks próbki

        Returns:
            Tuple zawierający:
                - input_window: Tensor (T_in, features)
                - target_window: Tensor (T_out, features)
        """
        traj_idx, start_idx = self.windows[idx]

        # Wycinanie okien z trajektorii
        input_end = start_idx + self.T_in
        target_end = input_end + self.T_out

        input_window = self.data[traj_idx, start_idx:input_end]
        target_window = self.data[traj_idx, input_end:target_end]

        return input_window, target_window

    def get_full_trajectory(self, traj_idx: int) -> torch.Tensor:
        """
        Pobiera pełną trajektorię.

        Args:
            traj_idx: Indeks trajektorii

        Returns:
            Tensor (num_steps, features)
        """
        return self.data[traj_idx]


class ConditionedTimeSeriesDataset(Dataset):
    """
    Dataset PyTorch dla szeregów czasowych z parametrami warunkującymi.

    Tworzy sliding windows z trajektorii, gdzie każda próbka
    składa się z okna wejściowego, parametrów warunkujących i okna docelowego.

    Używany dla modelu z parametryzacją bezwymiarową, gdzie sieć
    otrzymuje parametry (ζ, ω₀) jako dodatkowe wejście.

    Attributes:
        data: Znormalizowane dane trajektorii
        params: Parametry warunkujące dla każdej trajektorii
        T_in: Długość okna wejściowego
        T_out: Długość okna docelowego (horyzont predykcji)
        stride: Krok przesuwania okna
        windows: Lista indeksów (trajectory_idx, start_idx) dla każdego okna
    """

    def __init__(
        self,
        data: np.ndarray,
        params: np.ndarray,
        T_in: int,
        T_out: int,
        stride: int = 1
    ):
        """
        Inicjalizacja datasetu z parametrami warunkującymi.

        Args:
            data: Dane trajektorii, kształt (num_trajectories, num_steps, features)
            params: Parametry warunkujące, kształt (num_trajectories, param_size)
                    np. [zeta, omega_0] dla parametryzacji bezwymiarowej
            T_in: Długość okna wejściowego (historia)
            T_out: Długość okna docelowego (predykcja)
            stride: Krok przesuwania okna (domyślnie 1)
        """
        self.data = torch.FloatTensor(data)
        self.params = torch.FloatTensor(params)
        self.T_in = T_in
        self.T_out = T_out
        self.stride = stride

        # Wymiary danych
        self.num_trajectories = data.shape[0]
        self.num_steps = data.shape[1]
        self.num_features = data.shape[2]
        self.param_size = params.shape[1] if len(params.shape) > 1 else 1

        # Walidacja zgodności liczby trajektorii i parametrów
        if self.num_trajectories != len(params):
            raise ValueError(
                f"Liczba trajektorii ({self.num_trajectories}) "
                f"nie zgadza się z liczbą wektorów parametrów ({len(params)})"
            )

        # Minimalna długość trajektorii potrzebna dla jednego okna
        self.min_length = T_in + T_out

        # Tworzenie indeksów okien
        self.windows = self._create_windows()

    def _create_windows(self) -> List[Tuple[int, int]]:
        """
        Tworzy listę indeksów okien dla wszystkich trajektorii.

        Dla każdej trajektorii generuje wszystkie możliwe okna
        z zadanym krokiem (stride).

        Returns:
            Lista tupli (trajectory_idx, start_idx)
        """
        windows = []

        for traj_idx in range(self.num_trajectories):
            # Maksymalny indeks startowy dla tego okna
            max_start = self.num_steps - self.min_length

            # Generowanie indeksów z krokiem stride
            for start_idx in range(0, max_start + 1, self.stride):
                windows.append((traj_idx, start_idx))

        return windows

    def __len__(self) -> int:
        """Zwraca całkowitą liczbę okien w datasecie."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pobiera pojedynczą próbkę (okno wejściowe, parametry, okno docelowe).

        Args:
            idx: Indeks próbki

        Returns:
            Tuple zawierający:
                - input_window: Tensor (T_in, features)
                - params: Tensor (param_size,) - np. [zeta, omega_0]
                - target_window: Tensor (T_out, features)
        """
        traj_idx, start_idx = self.windows[idx]

        # Wycinanie okien z trajektorii
        input_end = start_idx + self.T_in
        target_end = input_end + self.T_out

        input_window = self.data[traj_idx, start_idx:input_end]
        target_window = self.data[traj_idx, input_end:target_end]

        # Parametry dla tej trajektorii
        params = self.params[traj_idx]

        return input_window, params, target_window

    def get_full_trajectory(self, traj_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pobiera pełną trajektorię wraz z parametrami.

        Args:
            traj_idx: Indeks trajektorii

        Returns:
            Tuple (trajectory, params)
        """
        return self.data[traj_idx], self.params[traj_idx]

    def get_params_stats(self) -> Dict[str, np.ndarray]:
        """
        Zwraca statystyki parametrów warunkujących.

        Returns:
            Słownik z min, max, mean, std dla każdego parametru
        """
        params_np = self.params.numpy()
        return {
            'min': params_np.min(axis=0),
            'max': params_np.max(axis=0),
            'mean': params_np.mean(axis=0),
            'std': params_np.std(axis=0)
        }


class TimeSeriesDataModule(pl.LightningDataModule):
    """
    Lightning DataModule dla szeregów czasowych.

    Zarządza:
    - Podziałem danych na train/val/test
    - Tworzeniem DataLoaderów
    - Konfiguracją batchowania

    Attributes:
        data: Pełne dane trajektorii (znormalizowane)
        T_in: Długość okna wejściowego
        T_out: Długość okna docelowego
        batch_size: Rozmiar batcha
        train_ratio: Proporcja danych treningowych
        val_ratio: Proporcja danych walidacyjnych
        num_workers: Liczba procesów do ładowania danych
    """

    def __init__(
        self,
        data: np.ndarray,
        T_in: int,
        T_out: int,
        batch_size: int = 64,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        stride: int = 1,
        num_workers: int = 0,
        seed: int = 42
    ):
        """
        Inicjalizacja DataModule.

        Args:
            data: Dane trajektorii, kształt (num_trajectories, num_steps, features)
            T_in: Długość okna wejściowego
            T_out: Długość okna docelowego
            batch_size: Rozmiar batcha
            train_ratio: Proporcja danych treningowych (0-1)
            val_ratio: Proporcja danych walidacyjnych (0-1)
            stride: Krok przesuwania okna
            num_workers: Liczba procesów do ładowania danych
            seed: Ziarno generatora losowego
        """
        super().__init__()

        self.data = data
        self.T_in = T_in
        self.T_out = T_out
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.stride = stride
        self.num_workers = num_workers
        self.seed = seed

        # Datasety (tworzone w setup())
        self.train_dataset: Optional[TimeSeriesDataset] = None
        self.val_dataset: Optional[TimeSeriesDataset] = None
        self.test_dataset: Optional[TimeSeriesDataset] = None

        # Walidacja proporcji
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("Suma proporcji musi wynosić 1.0")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Przygotowuje datasety dla różnych etapów.

        Dzieli trajektorie na zbiory train/val/test
        (podział na poziomie trajektorii, nie okien).

        Args:
            stage: Etap ('fit', 'validate', 'test', 'predict' lub None dla wszystkich)
        """
        # Liczba trajektorii
        num_trajectories = self.data.shape[0]

        # Losowe mieszanie indeksów trajektorii
        np.random.seed(self.seed)
        indices = np.random.permutation(num_trajectories)

        # Obliczenie granic podziału
        train_end = int(num_trajectories * self.train_ratio)
        val_end = train_end + int(num_trajectories * self.val_ratio)

        # Podział indeksów
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        # Tworzenie datasetów
        if stage == 'fit' or stage is None:
            self.train_dataset = TimeSeriesDataset(
                data=self.data[train_indices],
                T_in=self.T_in,
                T_out=self.T_out,
                stride=self.stride
            )
            self.val_dataset = TimeSeriesDataset(
                data=self.data[val_indices],
                T_in=self.T_in,
                T_out=self.T_out,
                stride=self.stride
            )

        if stage == 'test' or stage is None:
            self.test_dataset = TimeSeriesDataset(
                data=self.data[test_indices],
                T_in=self.T_in,
                T_out=self.T_out,
                stride=self.stride
            )

        if stage == 'predict' or stage is None:
            # Dla predykcji używamy danych testowych
            if self.test_dataset is None:
                self.test_dataset = TimeSeriesDataset(
                    data=self.data[test_indices],
                    T_in=self.T_in,
                    T_out=self.T_out,
                    stride=self.stride
                )

    def train_dataloader(self) -> DataLoader:
        """Zwraca DataLoader dla zbioru treningowego."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None
        )

    def val_dataloader(self) -> DataLoader:
        """Zwraca DataLoader dla zbioru walidacyjnego."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None
        )

    def test_dataloader(self) -> DataLoader:
        """Zwraca DataLoader dla zbioru testowego."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None
        )

    def predict_dataloader(self) -> DataLoader:
        """Zwraca DataLoader dla predykcji."""
        return self.test_dataloader()

    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zwraca przykładowy batch do testowania modelu.

        Returns:
            Tuple (input_batch, target_batch)
        """
        if self.train_dataset is None:
            self.setup('fit')

        loader = DataLoader(
            self.train_dataset,
            batch_size=min(self.batch_size, len(self.train_dataset)),
            shuffle=False
        )

        return next(iter(loader))

    def get_data_info(self) -> Dict:
        """
        Zwraca informacje o danych.

        Returns:
            Słownik z informacjami o wymiarach i liczbie próbek
        """
        if self.train_dataset is None:
            self.setup()

        return {
            'num_trajectories': self.data.shape[0],
            'trajectory_length': self.data.shape[1],
            'num_features': self.data.shape[2],
            'T_in': self.T_in,
            'T_out': self.T_out,
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'val_samples': len(self.val_dataset) if self.val_dataset else 0,
            'test_samples': len(self.test_dataset) if self.test_dataset else 0,
            'batch_size': self.batch_size
        }


class ConditionedTimeSeriesDataModule(pl.LightningDataModule):
    """
    Lightning DataModule dla szeregów czasowych z parametrami warunkującymi.

    Rozszerzenie TimeSeriesDataModule o obsługę parametrów warunkujących
    (np. ζ, ω₀ dla parametryzacji bezwymiarowej).

    Zarządza:
    - Podziałem danych na train/val/test
    - Normalizacją parametrów warunkujących
    - Tworzeniem DataLoaderów

    Attributes:
        data: Pełne dane trajektorii (znormalizowane)
        params: Parametry warunkujące dla każdej trajektorii
        T_in: Długość okna wejściowego
        T_out: Długość okna docelowego
        batch_size: Rozmiar batcha
        normalize_params: Czy normalizować parametry warunkujące
    """

    def __init__(
        self,
        data: np.ndarray,
        params: np.ndarray,
        T_in: int,
        T_out: int,
        batch_size: int = 64,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        stride: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        normalize_params: bool = True
    ):
        """
        Inicjalizacja DataModule z parametrami warunkującymi.

        Args:
            data: Dane trajektorii, kształt (num_trajectories, num_steps, features)
            params: Parametry warunkujące, kształt (num_trajectories, param_size)
            T_in: Długość okna wejściowego
            T_out: Długość okna docelowego
            batch_size: Rozmiar batcha
            train_ratio: Proporcja danych treningowych (0-1)
            val_ratio: Proporcja danych walidacyjnych (0-1)
            stride: Krok przesuwania okna
            num_workers: Liczba procesów do ładowania danych
            seed: Ziarno generatora losowego
            normalize_params: Czy normalizować parametry do [0, 1]
        """
        super().__init__()

        self.data = data
        self.params = params
        self.T_in = T_in
        self.T_out = T_out
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.stride = stride
        self.num_workers = num_workers
        self.seed = seed
        self.normalize_params = normalize_params

        # Statystyki normalizacji parametrów
        self.params_min: Optional[np.ndarray] = None
        self.params_max: Optional[np.ndarray] = None

        # Datasety (tworzone w setup())
        self.train_dataset: Optional[ConditionedTimeSeriesDataset] = None
        self.val_dataset: Optional[ConditionedTimeSeriesDataset] = None
        self.test_dataset: Optional[ConditionedTimeSeriesDataset] = None

        # Indeksy podziału (do odtworzenia)
        self.train_indices: Optional[np.ndarray] = None
        self.val_indices: Optional[np.ndarray] = None
        self.test_indices: Optional[np.ndarray] = None

        # Walidacja proporcji
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("Suma proporcji musi wynosić 1.0")

    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        """
        Normalizuje parametry do zakresu [0, 1].

        Args:
            params: Parametry do znormalizowania

        Returns:
            Znormalizowane parametry
        """
        if self.params_min is None or self.params_max is None:
            raise RuntimeError("Statystyki normalizacji nie zostały obliczone")

        # Unikanie dzielenia przez zero
        range_params = self.params_max - self.params_min
        range_params[range_params == 0] = 1.0

        return (params - self.params_min) / range_params

    def _denormalize_params(self, params: np.ndarray) -> np.ndarray:
        """
        Odwraca normalizację parametrów.

        Args:
            params: Znormalizowane parametry

        Returns:
            Parametry w oryginalnej skali
        """
        if self.params_min is None or self.params_max is None:
            raise RuntimeError("Statystyki normalizacji nie zostały obliczone")

        range_params = self.params_max - self.params_min
        return params * range_params + self.params_min

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Przygotowuje datasety dla różnych etapów.

        Dzieli trajektorie na zbiory train/val/test
        i oblicza statystyki normalizacji parametrów.

        Args:
            stage: Etap ('fit', 'validate', 'test', 'predict' lub None dla wszystkich)
        """
        # Liczba trajektorii
        num_trajectories = self.data.shape[0]

        # Losowe mieszanie indeksów trajektorii
        np.random.seed(self.seed)
        indices = np.random.permutation(num_trajectories)

        # Obliczenie granic podziału
        train_end = int(num_trajectories * self.train_ratio)
        val_end = train_end + int(num_trajectories * self.val_ratio)

        # Podział indeksów
        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]

        # Obliczenie statystyk normalizacji TYLKO na danych treningowych
        if self.normalize_params:
            train_params = self.params[self.train_indices]
            self.params_min = train_params.min(axis=0)
            self.params_max = train_params.max(axis=0)

            # Normalizacja wszystkich parametrów
            normalized_params = self._normalize_params(self.params)
        else:
            normalized_params = self.params

        # Tworzenie datasetów
        if stage == 'fit' or stage is None:
            self.train_dataset = ConditionedTimeSeriesDataset(
                data=self.data[self.train_indices],
                params=normalized_params[self.train_indices],
                T_in=self.T_in,
                T_out=self.T_out,
                stride=self.stride
            )
            self.val_dataset = ConditionedTimeSeriesDataset(
                data=self.data[self.val_indices],
                params=normalized_params[self.val_indices],
                T_in=self.T_in,
                T_out=self.T_out,
                stride=self.stride
            )

        if stage == 'test' or stage is None:
            self.test_dataset = ConditionedTimeSeriesDataset(
                data=self.data[self.test_indices],
                params=normalized_params[self.test_indices],
                T_in=self.T_in,
                T_out=self.T_out,
                stride=self.stride
            )

        if stage == 'predict' or stage is None:
            if self.test_dataset is None:
                self.test_dataset = ConditionedTimeSeriesDataset(
                    data=self.data[self.test_indices],
                    params=normalized_params[self.test_indices],
                    T_in=self.T_in,
                    T_out=self.T_out,
                    stride=self.stride
                )

    def train_dataloader(self) -> DataLoader:
        """Zwraca DataLoader dla zbioru treningowego."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None
        )

    def val_dataloader(self) -> DataLoader:
        """Zwraca DataLoader dla zbioru walidacyjnego."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None
        )

    def test_dataloader(self) -> DataLoader:
        """Zwraca DataLoader dla zbioru testowego."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None
        )

    def predict_dataloader(self) -> DataLoader:
        """Zwraca DataLoader dla predykcji."""
        return self.test_dataloader()

    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Zwraca przykładowy batch do testowania modelu.

        Returns:
            Tuple (input_batch, params_batch, target_batch)
        """
        if self.train_dataset is None:
            self.setup('fit')

        loader = DataLoader(
            self.train_dataset,
            batch_size=min(self.batch_size, len(self.train_dataset)),
            shuffle=False
        )

        return next(iter(loader))

    def get_data_info(self) -> Dict:
        """
        Zwraca informacje o danych.

        Returns:
            Słownik z informacjami o wymiarach i liczbie próbek
        """
        if self.train_dataset is None:
            self.setup()

        return {
            'num_trajectories': self.data.shape[0],
            'trajectory_length': self.data.shape[1],
            'num_features': self.data.shape[2],
            'param_size': self.params.shape[1] if len(self.params.shape) > 1 else 1,
            'T_in': self.T_in,
            'T_out': self.T_out,
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'val_samples': len(self.val_dataset) if self.val_dataset else 0,
            'test_samples': len(self.test_dataset) if self.test_dataset else 0,
            'batch_size': self.batch_size,
            'params_normalized': self.normalize_params
        }

    def get_params_normalization_stats(self) -> Dict[str, np.ndarray]:
        """
        Zwraca statystyki normalizacji parametrów.

        Returns:
            Słownik z min i max użytymi do normalizacji
        """
        return {
            'min': self.params_min,
            'max': self.params_max
        }

    def save_params_stats(self, filepath: str) -> None:
        """
        Zapisuje statystyki normalizacji parametrów do pliku.

        Args:
            filepath: Ścieżka do pliku
        """
        np.savez(
            filepath,
            params_min=self.params_min,
            params_max=self.params_max
        )

    def load_params_stats(self, filepath: str) -> None:
        """
        Wczytuje statystyki normalizacji parametrów z pliku.

        Args:
            filepath: Ścieżka do pliku
        """
        data = np.load(filepath)
        self.params_min = data['params_min']
        self.params_max = data['params_max']


if __name__ == "__main__":
    # Test modułu dataset
    from src.data_generation import generate_dataset
    from src.preprocessing import DataPreprocessor

    print("Test modułu TimeSeriesDataset i TimeSeriesDataModule")
    print("=" * 60)

    # Generowanie danych testowych
    dataset = generate_dataset(
        num_trajectories=100,
        dt=0.01,
        t_max=5.0,
        seed=42
    )

    trajectories = dataset['trajectories']
    print(f"\nWygenerowano {trajectories.shape[0]} trajektorii")
    print(f"Kształt danych: {trajectories.shape}")

    # Preprocessing
    preprocessor = DataPreprocessor(apply_filtering=False)
    normalized = preprocessor.fit_transform(trajectories)

    # Tworzenie DataModule
    T_in = 50
    T_out = 30

    datamodule = TimeSeriesDataModule(
        data=normalized,
        T_in=T_in,
        T_out=T_out,
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.15,
        stride=5,
        seed=42
    )

    # Setup
    datamodule.setup()

    # Informacje o danych
    info = datamodule.get_data_info()
    print("\nInformacje o DataModule:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test batcha
    input_batch, target_batch = datamodule.get_sample_batch()
    print(f"\nKształty batcha:")
    print(f"  Input: {input_batch.shape}")
    print(f"  Target: {target_batch.shape}")

    # Sprawdzenie typów danych
    print(f"\nTypy danych:")
    print(f"  Input dtype: {input_batch.dtype}")
    print(f"  Target dtype: {target_batch.dtype}")

    # Test DataLoaderów
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    print(f"\nLiczba batchy:")
    print(f"  Train: {len(train_loader)}")
    print(f"  Val: {len(val_loader)}")
    print(f"  Test: {len(test_loader)}")

    # Iteracja przez jeden batch
    for batch_idx, (inp, tgt) in enumerate(train_loader):
        print(f"\nPrzykładowy batch {batch_idx}:")
        print(f"  Input shape: {inp.shape}")
        print(f"  Target shape: {tgt.shape}")
        break

    print("\n" + "=" * 60)
    print("Test zakończony pomyślnie!")
