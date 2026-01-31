"""
Moduł metryk ewaluacji.

Zawiera funkcje do obliczania jakości predykcji modelu,
w tym metryki punktowe i probabilistyczne.
"""

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


def compute_rmse(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    per_feature: bool = False
) -> Union[float, np.ndarray]:
    """
    Oblicza Root Mean Square Error (RMSE).

    RMSE = sqrt(mean((predictions - targets)^2))

    Args:
        predictions: Predykcje modelu, kształt (..., features)
        targets: Wartości docelowe, kształt (..., features)
        per_feature: Czy zwracać RMSE osobno dla każdej cechy

    Returns:
        RMSE (skalar lub wektor dla per_feature=True)
    """
    # Konwersja do numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    squared_errors = (predictions - targets) ** 2

    if per_feature:
        # RMSE osobno dla każdej cechy (ostatni wymiar)
        # Spłaszczamy wszystkie wymiary oprócz ostatniego
        flat = squared_errors.reshape(-1, squared_errors.shape[-1])
        return np.sqrt(flat.mean(axis=0))
    else:
        return float(np.sqrt(squared_errors.mean()))


def compute_mae(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    per_feature: bool = False
) -> Union[float, np.ndarray]:
    """
    Oblicza Mean Absolute Error (MAE).

    MAE = mean(|predictions - targets|)

    Args:
        predictions: Predykcje modelu, kształt (..., features)
        targets: Wartości docelowe, kształt (..., features)
        per_feature: Czy zwracać MAE osobno dla każdej cechy

    Returns:
        MAE (skalar lub wektor dla per_feature=True)
    """
    # Konwersja do numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    absolute_errors = np.abs(predictions - targets)

    if per_feature:
        flat = absolute_errors.reshape(-1, absolute_errors.shape[-1])
        return flat.mean(axis=0)
    else:
        return float(absolute_errors.mean())


def compute_nll(
    mu: Union[np.ndarray, torch.Tensor],
    sigma: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    per_feature: bool = False,
    eps: float = 1e-6
) -> Union[float, np.ndarray]:
    """
    Oblicza Negative Log-Likelihood dla rozkładu Gaussowskiego.

    NLL = 0.5 * (log(2π) + 2*log(σ) + ((x-μ)/σ)^2)

    Args:
        mu: Predykowane średnie, kształt (..., features)
        sigma: Predykowane odchylenia standardowe, kształt (..., features)
        targets: Wartości docelowe, kształt (..., features)
        per_feature: Czy zwracać NLL osobno dla każdej cechy
        eps: Mała wartość dla stabilności numerycznej

    Returns:
        NLL (skalar lub wektor dla per_feature=True)
    """
    # Konwersja do numpy
    if isinstance(mu, torch.Tensor):
        mu = mu.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Zabezpieczenie przed sigma = 0
    sigma = np.maximum(sigma, eps)

    # Obliczenie NLL
    log_2pi = math.log(2 * math.pi)
    squared_error = ((targets - mu) / sigma) ** 2
    log_sigma = np.log(sigma)

    nll = 0.5 * (log_2pi + 2 * log_sigma + squared_error)

    if per_feature:
        flat = nll.reshape(-1, nll.shape[-1])
        return flat.mean(axis=0)
    else:
        return float(nll.mean())


def compute_coverage_probability(
    mu: Union[np.ndarray, torch.Tensor],
    sigma: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    confidence_levels: Optional[list] = None
) -> Dict[str, float]:
    """
    Oblicza prawdopodobieństwo pokrycia dla przedziałów ufności.

    Sprawdza, jaki procent prawdziwych wartości mieści się
    w zadanych przedziałach ufności (np. ±1σ, ±2σ).

    Args:
        mu: Predykowane średnie, kształt (..., features)
        sigma: Predykowane odchylenia standardowe, kształt (..., features)
        targets: Wartości docelowe, kształt (..., features)
        confidence_levels: Lista poziomów ufności w sigmas (domyślnie [1, 2, 3])

    Returns:
        Słownik z prawdopodobieństwami pokrycia dla każdego poziomu
    """
    if confidence_levels is None:
        confidence_levels = [1.0, 2.0, 3.0]

    # Konwersja do numpy
    if isinstance(mu, torch.Tensor):
        mu = mu.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Spłaszczenie do 1D dla obliczenia globalnego pokrycia
    mu_flat = mu.flatten()
    sigma_flat = sigma.flatten()
    targets_flat = targets.flatten()

    coverage = {}

    for level in confidence_levels:
        # Granice przedziału
        lower = mu_flat - level * sigma_flat
        upper = mu_flat + level * sigma_flat

        # Sprawdzenie czy target mieści się w przedziale
        in_interval = (targets_flat >= lower) & (targets_flat <= upper)
        prob = float(in_interval.mean())

        # Teoretyczne pokrycie dla rozkładu normalnego
        # P(|X - μ| < k*σ) dla różnych k
        theoretical = {
            1.0: 0.6827,  # 68.27%
            2.0: 0.9545,  # 95.45%
            3.0: 0.9973   # 99.73%
        }

        key = f"coverage_{level:.1f}sigma"
        coverage[key] = prob

        if level in theoretical:
            coverage[f"expected_{level:.1f}sigma"] = theoretical[level]

    return coverage


def compute_calibration_error(
    mu: Union[np.ndarray, torch.Tensor],
    sigma: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Oblicza błąd kalibracji rozkładu predykcji.

    Dla dobrze skalibrowanego modelu probabilistycznego,
    x% wartości powinno mieścić się w przedziale ufności x%.

    Args:
        mu: Predykowane średnie
        sigma: Predykowane odchylenia standardowe
        targets: Wartości docelowe
        num_bins: Liczba binów do sprawdzenia

    Returns:
        Słownik z metrykami kalibracji
    """
    # Konwersja do numpy
    if isinstance(mu, torch.Tensor):
        mu = mu.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Spłaszczenie
    mu_flat = mu.flatten()
    sigma_flat = sigma.flatten()
    targets_flat = targets.flatten()

    # Obliczenie z-scores
    z_scores = np.abs((targets_flat - mu_flat) / np.maximum(sigma_flat, 1e-6))

    # Teoretyczne kwantyle dla |Z| (rozkład półnormalny)
    from scipy import stats
    expected_coverage = np.linspace(0.1, 0.9, num_bins)
    z_thresholds = stats.norm.ppf((1 + expected_coverage) / 2)

    # Faktyczne pokrycie
    actual_coverage = []
    for z_thresh in z_thresholds:
        actual_coverage.append((z_scores <= z_thresh).mean())

    actual_coverage = np.array(actual_coverage)

    # Błąd kalibracji
    calibration_errors = actual_coverage - expected_coverage

    return {
        'mean_calibration_error': float(np.abs(calibration_errors).mean()),
        'max_calibration_error': float(np.abs(calibration_errors).max()),
        'expected_coverage': expected_coverage.tolist(),
        'actual_coverage': actual_coverage.tolist()
    }


def compute_crps(
    mu: Union[np.ndarray, torch.Tensor],
    sigma: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Oblicza Continuous Ranked Probability Score (CRPS).

    CRPS jest metryką łączącą kalibrację i precyzję.
    Mniejsze wartości są lepsze.

    Dla rozkładu normalnego istnieje zamknięta formuła.

    Args:
        mu: Predykowane średnie
        sigma: Predykowane odchylenia standardowe
        targets: Wartości docelowe

    Returns:
        Średni CRPS
    """
    from scipy import stats

    # Konwersja do numpy
    if isinstance(mu, torch.Tensor):
        mu = mu.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Spłaszczenie
    mu_flat = mu.flatten()
    sigma_flat = sigma.flatten()
    targets_flat = targets.flatten()

    # Standaryzacja
    z = (targets_flat - mu_flat) / np.maximum(sigma_flat, 1e-6)

    # CRPS dla rozkładu normalnego
    # CRPS = σ * (z * (2*Φ(z) - 1) + 2*φ(z) - 1/√π)
    phi_z = stats.norm.cdf(z)
    pdf_z = stats.norm.pdf(z)

    crps = sigma_flat * (z * (2 * phi_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))

    return float(crps.mean())


def compute_all_metrics(
    mu: Union[np.ndarray, torch.Tensor],
    sigma: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    feature_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Oblicza wszystkie metryki ewaluacji.

    Args:
        mu: Predykowane średnie, kształt (batch, T_out, features)
        sigma: Predykowane odchylenia std, kształt (batch, T_out, features)
        targets: Wartości docelowe, kształt (batch, T_out, features)
        feature_names: Opcjonalne nazwy cech (np. ['x', 'v'])

    Returns:
        Słownik ze wszystkimi metrykami
    """
    if feature_names is None:
        feature_names = ['x', 'v']

    metrics = {}

    # Metryki globalne
    metrics['rmse'] = compute_rmse(mu, targets)
    metrics['mae'] = compute_mae(mu, targets)
    metrics['nll'] = compute_nll(mu, sigma, targets)
    metrics['crps'] = compute_crps(mu, sigma, targets)

    # Metryki per feature
    rmse_per_feature = compute_rmse(mu, targets, per_feature=True)
    mae_per_feature = compute_mae(mu, targets, per_feature=True)
    nll_per_feature = compute_nll(mu, sigma, targets, per_feature=True)

    for i, name in enumerate(feature_names):
        metrics[f'rmse_{name}'] = float(rmse_per_feature[i])
        metrics[f'mae_{name}'] = float(mae_per_feature[i])
        metrics[f'nll_{name}'] = float(nll_per_feature[i])

    # Coverage probability
    coverage = compute_coverage_probability(mu, sigma, targets)
    metrics.update(coverage)

    # Calibration
    calibration = compute_calibration_error(mu, sigma, targets)
    metrics['mean_calibration_error'] = calibration['mean_calibration_error']

    return metrics


if __name__ == "__main__":
    # Test modułu metrics
    print("Test metryk ewaluacji")
    print("=" * 60)

    # Symulacja predykcji
    np.random.seed(42)
    batch_size = 100
    T_out = 30
    features = 2

    # Generowanie danych testowych
    # "Idealne" predykcje z dodanym szumem
    targets = np.random.randn(batch_size, T_out, features)
    noise_mu = np.random.randn(batch_size, T_out, features) * 0.3
    mu = targets + noise_mu  # Predykcje z błędem
    sigma = np.abs(np.random.randn(batch_size, T_out, features) * 0.5) + 0.1

    # Test RMSE
    print("\n1. Test compute_rmse:")
    rmse = compute_rmse(mu, targets)
    rmse_per_feature = compute_rmse(mu, targets, per_feature=True)
    print(f"   RMSE (global): {rmse:.4f}")
    print(f"   RMSE (per feature): {rmse_per_feature}")

    # Test MAE
    print("\n2. Test compute_mae:")
    mae = compute_mae(mu, targets)
    mae_per_feature = compute_mae(mu, targets, per_feature=True)
    print(f"   MAE (global): {mae:.4f}")
    print(f"   MAE (per feature): {mae_per_feature}")

    # Test NLL
    print("\n3. Test compute_nll:")
    nll = compute_nll(mu, sigma, targets)
    print(f"   NLL: {nll:.4f}")

    # Test coverage probability
    print("\n4. Test compute_coverage_probability:")
    coverage = compute_coverage_probability(mu, sigma, targets)
    for key, value in coverage.items():
        print(f"   {key}: {value:.4f}")

    # Test kalibracji
    print("\n5. Test compute_calibration_error:")
    calibration = compute_calibration_error(mu, sigma, targets)
    print(f"   Mean calibration error: {calibration['mean_calibration_error']:.4f}")
    print(f"   Max calibration error: {calibration['max_calibration_error']:.4f}")

    # Test CRPS
    print("\n6. Test compute_crps:")
    crps = compute_crps(mu, sigma, targets)
    print(f"   CRPS: {crps:.4f}")

    # Test wszystkich metryk
    print("\n7. Test compute_all_metrics:")
    all_metrics = compute_all_metrics(mu, sigma, targets, ['x', 'v'])
    print("   Wszystkie metryki:")
    for key, value in all_metrics.items():
        if isinstance(value, float):
            print(f"     {key}: {value:.4f}")

    # Test z tensorami PyTorch
    print("\n8. Test z tensorami PyTorch:")
    mu_torch = torch.tensor(mu)
    sigma_torch = torch.tensor(sigma)
    targets_torch = torch.tensor(targets)

    rmse_torch = compute_rmse(mu_torch, targets_torch)
    print(f"   RMSE (torch): {rmse_torch:.4f}")

    print("\n" + "=" * 60)
    print("Test zakończony pomyślnie!")
