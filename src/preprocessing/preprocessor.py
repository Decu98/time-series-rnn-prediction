"""
Moduł preprocessingu danych czasowych.

Zawiera funkcje do:
- Resamplingu sygnałów do stałego kroku czasowego
- Normalizacji i denormalizacji cech
- Filtrowania sygnałów (Savitzky-Golay)
- Klasy opakowującej DataPreprocessor
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


@dataclass
class NormalizationStats:
    """
    Statystyki normalizacji dla poszczególnych cech.

    Przechowuje średnie i odchylenia standardowe osobno
    dla pozycji (x) i prędkości (v).

    Attributes:
        mean_x: Średnia położenia
        std_x: Odchylenie standardowe położenia
        mean_v: Średnia prędkości
        std_v: Odchylenie standardowe prędkości
    """
    mean_x: float
    std_x: float
    mean_v: float
    std_v: float

    def to_dict(self) -> Dict[str, float]:
        """Konwertuje statystyki do słownika."""
        return {
            'mean_x': self.mean_x,
            'std_x': self.std_x,
            'mean_v': self.mean_v,
            'std_v': self.std_v
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'NormalizationStats':
        """Tworzy obiekt ze słownika."""
        return cls(
            mean_x=d['mean_x'],
            std_x=d['std_x'],
            mean_v=d['mean_v'],
            std_v=d['std_v']
        )


def resample_to_constant_dt(
    time: np.ndarray,
    data: np.ndarray,
    target_dt: float,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resampluje dane do stałego kroku czasowego.

    Używa interpolacji do przekształcenia nieregularnie
    próbkowanych danych na regularną siatkę czasową.

    Args:
        time: Oryginalny wektor czasu (może być nieregularny)
        data: Dane do resamplingu, kształt (n_samples,) lub (n_samples, n_features)
        target_dt: Docelowy krok czasowy [s]
        method: Metoda interpolacji ('linear', 'cubic', 'quadratic')

    Returns:
        Tuple zawierający:
            - new_time: Nowy wektor czasu z regularnym krokiem
            - new_data: Zresamplowane dane
    """
    # Tworzenie nowej siatki czasowej
    t_start = time[0]
    t_end = time[-1]
    new_time = np.arange(t_start, t_end, target_dt)

    # Sprawdzenie czy dane są 1D czy 2D
    if data.ndim == 1:
        interpolator = interp1d(time, data, kind=method, fill_value='extrapolate')
        new_data = interpolator(new_time)
    else:
        # Interpolacja każdej cechy osobno
        n_features = data.shape[1]
        new_data = np.zeros((len(new_time), n_features))

        for i in range(n_features):
            interpolator = interp1d(
                time, data[:, i],
                kind=method,
                fill_value='extrapolate'
            )
            new_data[:, i] = interpolator(new_time)

    return new_time, new_data


def normalize_features(
    data: np.ndarray,
    stats: Optional[NormalizationStats] = None
) -> Tuple[np.ndarray, NormalizationStats]:
    """
    Normalizuje cechy (standaryzacja z-score).

    Normalizacja jest przeprowadzana osobno dla położenia (x)
    i prędkości (v), aby zachować ich różne skale fizyczne.

    Args:
        data: Dane wejściowe, kształt (..., 2) gdzie ostatni wymiar to [x, v]
        stats: Opcjonalne statystyki normalizacji (jeśli None, obliczane z danych)

    Returns:
        Tuple zawierający:
            - normalized_data: Znormalizowane dane
            - stats: Statystyki użyte do normalizacji
    """
    # Spłaszczenie do 2D dla obliczenia statystyk
    original_shape = data.shape
    flat_data = data.reshape(-1, 2)

    if stats is None:
        # Obliczenie statystyk z danych treningowych
        stats = NormalizationStats(
            mean_x=float(np.mean(flat_data[:, 0])),
            std_x=float(np.std(flat_data[:, 0])),
            mean_v=float(np.mean(flat_data[:, 1])),
            std_v=float(np.std(flat_data[:, 1]))
        )

        # Zabezpieczenie przed dzieleniem przez zero
        if stats.std_x < 1e-8:
            stats.std_x = 1.0
        if stats.std_v < 1e-8:
            stats.std_v = 1.0

    # Normalizacja
    normalized = np.zeros_like(flat_data)
    normalized[:, 0] = (flat_data[:, 0] - stats.mean_x) / stats.std_x
    normalized[:, 1] = (flat_data[:, 1] - stats.mean_v) / stats.std_v

    # Przywrócenie oryginalnego kształtu
    normalized = normalized.reshape(original_shape)

    return normalized, stats


def denormalize_features(
    data: np.ndarray,
    stats: NormalizationStats
) -> np.ndarray:
    """
    Denormalizuje cechy (odwrócenie standaryzacji z-score).

    Args:
        data: Znormalizowane dane, kształt (..., 2)
        stats: Statystyki użyte do normalizacji

    Returns:
        Zdenormalizowane dane w oryginalnej skali
    """
    original_shape = data.shape
    flat_data = data.reshape(-1, 2)

    denormalized = np.zeros_like(flat_data)
    denormalized[:, 0] = flat_data[:, 0] * stats.std_x + stats.mean_x
    denormalized[:, 1] = flat_data[:, 1] * stats.std_v + stats.mean_v

    return denormalized.reshape(original_shape)


def denormalize_gaussian_params(
    mu: np.ndarray,
    sigma: np.ndarray,
    stats: NormalizationStats
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Denormalizuje parametry rozkładu Gaussowskiego (μ, σ).

    Dla rozkładu normalnego:
        - μ_denorm = μ_norm * std + mean
        - σ_denorm = σ_norm * std

    Args:
        mu: Średnie (znormalizowane), kształt (..., 2)
        sigma: Odchylenia standardowe (znormalizowane), kształt (..., 2)
        stats: Statystyki normalizacji

    Returns:
        Tuple zawierający:
            - mu_denorm: Zdenormalizowane średnie
            - sigma_denorm: Zdenormalizowane odchylenia standardowe
    """
    original_shape = mu.shape
    flat_mu = mu.reshape(-1, 2)
    flat_sigma = sigma.reshape(-1, 2)

    # Denormalizacja średnich
    mu_denorm = np.zeros_like(flat_mu)
    mu_denorm[:, 0] = flat_mu[:, 0] * stats.std_x + stats.mean_x
    mu_denorm[:, 1] = flat_mu[:, 1] * stats.std_v + stats.mean_v

    # Denormalizacja odchyleń standardowych (tylko skalowanie)
    sigma_denorm = np.zeros_like(flat_sigma)
    sigma_denorm[:, 0] = flat_sigma[:, 0] * stats.std_x
    sigma_denorm[:, 1] = flat_sigma[:, 1] * stats.std_v

    return mu_denorm.reshape(original_shape), sigma_denorm.reshape(original_shape)


def apply_filter(
    data: np.ndarray,
    window_length: int = 5,
    polyorder: int = 2,
    axis: int = 0
) -> np.ndarray:
    """
    Stosuje filtr Savitzky-Golay do wygładzania danych.

    Filtr ten zachowuje kształt sygnału lepiej niż proste
    uśrednianie, szczególnie dla danych z ostrymi zmianami.

    Args:
        data: Dane wejściowe
        window_length: Długość okna filtra (musi być nieparzysta)
        polyorder: Rząd wielomianu dopasowania
        axis: Oś wzdłuż której filtrować

    Returns:
        Przefiltrowane dane
    """
    # Zapewnienie nieparzystej długości okna
    if window_length % 2 == 0:
        window_length += 1

    # Sprawdzenie czy okno nie jest za duże
    if window_length > data.shape[axis]:
        window_length = data.shape[axis]
        if window_length % 2 == 0:
            window_length -= 1

    # Sprawdzenie minimalnej długości
    if window_length < polyorder + 2:
        return data

    return savgol_filter(
        data,
        window_length=window_length,
        polyorder=polyorder,
        axis=axis
    )


class DataPreprocessor:
    """
    Klasa opakowująca preprocessing danych czasowych.

    Agreguje wszystkie operacje preprocessingu i przechowuje
    statystyki normalizacji potrzebne do denormalizacji predykcji.

    Attributes:
        stats: Statystyki normalizacji (ustawiane po fit)
        apply_filtering: Czy stosować filtrowanie
        filter_window: Rozmiar okna filtru
        filter_order: Rząd wielomianu filtru
        is_fitted: Czy preprocessor został dopasowany do danych
    """

    def __init__(
        self,
        apply_filtering: bool = False,
        filter_window: int = 5,
        filter_order: int = 2
    ):
        """
        Inicjalizacja preprocessora.

        Args:
            apply_filtering: Czy stosować filtrowanie Savitzky-Golay
            filter_window: Rozmiar okna filtru
            filter_order: Rząd wielomianu filtru
        """
        self.stats: Optional[NormalizationStats] = None
        self.apply_filtering = apply_filtering
        self.filter_window = filter_window
        self.filter_order = filter_order
        self.is_fitted = False

    def fit(self, data: np.ndarray) -> 'DataPreprocessor':
        """
        Dopasowuje preprocessor do danych treningowych.

        Oblicza statystyki normalizacji na podstawie danych.

        Args:
            data: Dane treningowe, kształt (..., 2)

        Returns:
            self (dla chainingu)
        """
        # Opcjonalne filtrowanie przed obliczeniem statystyk
        if self.apply_filtering:
            processed_data = self._apply_filter_to_trajectories(data)
        else:
            processed_data = data

        # Obliczenie statystyk normalizacji
        _, self.stats = normalize_features(processed_data)
        self.is_fitted = True

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transformuje dane używając dopasowanych statystyk.

        Args:
            data: Dane do transformacji, kształt (..., 2)

        Returns:
            Przetworzone (opcjonalnie przefiltrowane i znormalizowane) dane

        Raises:
            RuntimeError: Jeśli preprocessor nie został dopasowany
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor nie został dopasowany. Użyj fit() przed transform()."
            )

        # Opcjonalne filtrowanie
        if self.apply_filtering:
            processed_data = self._apply_filter_to_trajectories(data)
        else:
            processed_data = data.copy()

        # Normalizacja
        normalized, _ = normalize_features(processed_data, self.stats)

        return normalized

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Dopasowuje preprocessor i transformuje dane w jednym kroku.

        Args:
            data: Dane treningowe, kształt (..., 2)

        Returns:
            Przetworzone dane
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Odwraca transformację (denormalizacja).

        Args:
            data: Znormalizowane dane, kształt (..., 2)

        Returns:
            Zdenormalizowane dane

        Raises:
            RuntimeError: Jeśli preprocessor nie został dopasowany
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor nie został dopasowany. Użyj fit() przed inverse_transform()."
            )

        return denormalize_features(data, self.stats)

    def inverse_transform_gaussian(
        self,
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Denormalizuje parametry rozkładu Gaussowskiego.

        Args:
            mu: Znormalizowane średnie, kształt (..., 2)
            sigma: Znormalizowane odchylenia standardowe, kształt (..., 2)

        Returns:
            Tuple zawierający zdenormalizowane (mu, sigma)

        Raises:
            RuntimeError: Jeśli preprocessor nie został dopasowany
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor nie został dopasowany."
            )

        return denormalize_gaussian_params(mu, sigma, self.stats)

    def _apply_filter_to_trajectories(self, data: np.ndarray) -> np.ndarray:
        """
        Stosuje filtr do trajektorii.

        Obsługuje zarówno pojedyncze trajektorie (2D) jak i
        batch trajektorii (3D).

        Args:
            data: Dane wejściowe, kształt (T, 2) lub (N, T, 2)

        Returns:
            Przefiltrowane dane
        """
        if data.ndim == 2:
            # Pojedyncza trajektoria (T, 2)
            return apply_filter(
                data,
                window_length=self.filter_window,
                polyorder=self.filter_order,
                axis=0
            )
        elif data.ndim == 3:
            # Batch trajektorii (N, T, 2)
            filtered = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered[i] = apply_filter(
                    data[i],
                    window_length=self.filter_window,
                    polyorder=self.filter_order,
                    axis=0
                )
            return filtered
        else:
            raise ValueError(f"Nieobsługiwany kształt danych: {data.shape}")

    def save_stats(self, filepath: str) -> None:
        """
        Zapisuje statystyki normalizacji do pliku.

        Args:
            filepath: Ścieżka do pliku wyjściowego
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor nie został dopasowany.")

        np.savez(
            filepath,
            **self.stats.to_dict(),
            apply_filtering=self.apply_filtering,
            filter_window=self.filter_window,
            filter_order=self.filter_order
        )

    def load_stats(self, filepath: str) -> 'DataPreprocessor':
        """
        Wczytuje statystyki normalizacji z pliku.

        Args:
            filepath: Ścieżka do pliku wejściowego

        Returns:
            self (dla chainingu)
        """
        data = np.load(filepath)

        self.stats = NormalizationStats(
            mean_x=float(data['mean_x']),
            std_x=float(data['std_x']),
            mean_v=float(data['mean_v']),
            std_v=float(data['std_v'])
        )

        self.apply_filtering = bool(data['apply_filtering'])
        self.filter_window = int(data['filter_window'])
        self.filter_order = int(data['filter_order'])
        self.is_fitted = True

        return self


if __name__ == "__main__":
    # Test modułu preprocessingu
    from src.data_generation import generate_dataset

    # Generowanie danych testowych
    dataset = generate_dataset(
        num_trajectories=10,
        dt=0.01,
        t_max=5.0,
        noise_std=0.05,
        seed=42
    )

    trajectories = dataset['trajectories']
    print(f"Kształt danych wejściowych: {trajectories.shape}")

    # Test preprocessora
    preprocessor = DataPreprocessor(
        apply_filtering=True,
        filter_window=7,
        filter_order=2
    )

    # Fit i transform
    normalized = preprocessor.fit_transform(trajectories)

    print(f"\nStatystyki normalizacji:")
    print(f"  mean_x: {preprocessor.stats.mean_x:.4f}")
    print(f"  std_x: {preprocessor.stats.std_x:.4f}")
    print(f"  mean_v: {preprocessor.stats.mean_v:.4f}")
    print(f"  std_v: {preprocessor.stats.std_v:.4f}")

    print(f"\nPo normalizacji:")
    flat_normalized = normalized.reshape(-1, 2)
    print(f"  mean_x: {np.mean(flat_normalized[:, 0]):.6f}")
    print(f"  std_x: {np.std(flat_normalized[:, 0]):.6f}")
    print(f"  mean_v: {np.mean(flat_normalized[:, 1]):.6f}")
    print(f"  std_v: {np.std(flat_normalized[:, 1]):.6f}")

    # Test denormalizacji
    denormalized = preprocessor.inverse_transform(normalized)
    reconstruction_error = np.mean(np.abs(trajectories - denormalized))
    print(f"\nBłąd rekonstrukcji (po filtrze): {reconstruction_error:.6f}")
