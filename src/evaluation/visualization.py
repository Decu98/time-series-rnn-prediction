"""
Moduł wizualizacji wyników.

Zawiera funkcje do tworzenia wykresów prezentujących:
- Predykcje z przedziałami ufności
- Krzywe uczenia (loss, metryki)
- Portrety fazowe
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_prediction_with_uncertainty(
    time_input: np.ndarray,
    time_output: np.ndarray,
    input_seq: np.ndarray,
    target_seq: np.ndarray,
    mu_seq: np.ndarray,
    sigma_seq: np.ndarray,
    feature_idx: int = 0,
    feature_name: str = 'x',
    confidence_levels: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Tworzy wykres predykcji z przedziałami ufności.

    Wyświetla:
    - Sekwencję wejściową (historia)
    - Wartości docelowe (ground truth)
    - Predykowaną średnią
    - Przedziały ufności (±1σ, ±2σ)

    Args:
        time_input: Wektor czasu dla danych wejściowych
        time_output: Wektor czasu dla predykcji
        input_seq: Sekwencja wejściowa (T_in, features)
        target_seq: Sekwencja docelowa (T_out, features)
        mu_seq: Predykowana średnia (T_out, features)
        sigma_seq: Predykowane odchylenie std (T_out, features)
        feature_idx: Indeks cechy do wizualizacji (0 dla x, 1 dla v)
        feature_name: Nazwa cechy do etykiet
        confidence_levels: Lista poziomów ufności w sigmas (domyślnie [1, 2])
        figsize: Rozmiar figury
        title: Opcjonalny tytuł wykresu
        save_path: Opcjonalna ścieżka do zapisu wykresu

    Returns:
        Obiekt Figure matplotlib
    """
    if confidence_levels is None:
        confidence_levels = [1.0, 2.0]

    # Konwersja tensorów do numpy
    if isinstance(input_seq, torch.Tensor):
        input_seq = input_seq.detach().cpu().numpy()
    if isinstance(target_seq, torch.Tensor):
        target_seq = target_seq.detach().cpu().numpy()
    if isinstance(mu_seq, torch.Tensor):
        mu_seq = mu_seq.detach().cpu().numpy()
    if isinstance(sigma_seq, torch.Tensor):
        sigma_seq = sigma_seq.detach().cpu().numpy()

    # Wyciągnięcie wybranej cechy
    input_values = input_seq[:, feature_idx]
    target_values = target_seq[:, feature_idx]
    mu_values = mu_seq[:, feature_idx]
    sigma_values = sigma_seq[:, feature_idx]

    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=figsize)

    # Kolory dla przedziałów ufności
    colors = plt.cm.Blues(np.linspace(0.2, 0.6, len(confidence_levels)))

    # Rysowanie przedziałów ufności (od największego do najmniejszego)
    for level, color in zip(sorted(confidence_levels, reverse=True), colors):
        lower = mu_values - level * sigma_values
        upper = mu_values + level * sigma_values
        ax.fill_between(
            time_output, lower, upper,
            alpha=0.3,
            color=color,
            label=f'±{level:.0f}σ ({level*68.27:.0f}% CI)' if level == 1 else f'±{level:.0f}σ ({95.45 if level==2 else 99.73:.0f}% CI)'
        )

    # Dane wejściowe (historia)
    ax.plot(
        time_input, input_values,
        'b-', linewidth=2,
        label='Dane wejściowe'
    )

    # Wartości docelowe (ground truth)
    ax.plot(
        time_output, target_values,
        'g-', linewidth=2,
        label='Wartości rzeczywiste'
    )

    # Predykowana średnia
    ax.plot(
        time_output, mu_values,
        'r--', linewidth=2,
        label='Predykcja (μ)'
    )

    # Linia rozdzielająca wejście od predykcji
    ax.axvline(
        x=time_input[-1],
        color='gray',
        linestyle=':',
        linewidth=1.5,
        alpha=0.7
    )

    # Formatowanie
    ax.set_xlabel('Czas [s]', fontsize=12)
    ax.set_ylabel(f'{feature_name}', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Predykcja {feature_name} z przedziałami ufności', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_rmse: Optional[List[float]] = None,
    val_rmse: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 5),
    title: str = 'Krzywe uczenia',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Tworzy wykres krzywych uczenia.

    Wyświetla:
    - Strata treningowa i walidacyjna
    - Opcjonalnie: RMSE treningowe i walidacyjne

    Args:
        train_losses: Lista strat treningowych (per epoka)
        val_losses: Lista strat walidacyjnych (per epoka)
        train_rmse: Opcjonalna lista RMSE treningowego
        val_rmse: Opcjonalna lista RMSE walidacyjnego
        figsize: Rozmiar figury
        title: Tytuł wykresu
        save_path: Opcjonalna ścieżka do zapisu

    Returns:
        Obiekt Figure matplotlib
    """
    epochs = range(1, len(train_losses) + 1)

    # Liczba subplotów
    num_plots = 1 if train_rmse is None else 2

    fig, axes = plt.subplots(1, num_plots, figsize=figsize)

    if num_plots == 1:
        axes = [axes]

    # Wykres straty
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss')
    axes[0].set_xlabel('Epoka', fontsize=12)
    axes[0].set_ylabel('Strata (NLL)', fontsize=12)
    axes[0].set_title('Funkcja straty', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Opcjonalny wykres RMSE
    if train_rmse is not None and num_plots > 1:
        axes[1].plot(epochs, train_rmse, 'b-', linewidth=2, label='Train RMSE')
        axes[1].plot(epochs, val_rmse, 'r-', linewidth=2, label='Val RMSE')
        axes[1].set_xlabel('Epoka', fontsize=12)
        axes[1].set_ylabel('RMSE', fontsize=12)
        axes[1].set_title('Root Mean Square Error', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_phase_space(
    trajectories: List[np.ndarray],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 8),
    title: str = 'Portret fazowy',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Tworzy wykres portretu fazowego (x vs v).

    Args:
        trajectories: Lista trajektorii [(T, 2), ...]
        labels: Opcjonalne etykiety dla trajektorii
        colors: Opcjonalne kolory dla trajektorii
        figsize: Rozmiar figury
        title: Tytuł wykresu
        save_path: Opcjonalna ścieżka do zapisu

    Returns:
        Obiekt Figure matplotlib
    """
    if labels is None:
        labels = [f'Trajektoria {i+1}' for i in range(len(trajectories))]

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    fig, ax = plt.subplots(figsize=figsize)

    for traj, label, color in zip(trajectories, labels, colors):
        # Konwersja do numpy
        if isinstance(traj, torch.Tensor):
            traj = traj.detach().cpu().numpy()

        x = traj[:, 0]
        v = traj[:, 1]

        # Rysowanie trajektorii
        ax.plot(x, v, color=color, linewidth=1.5, label=label, alpha=0.8)

        # Oznaczenie punktu startowego
        ax.scatter(x[0], v[0], color=color, s=50, marker='o', zorder=5)

        # Oznaczenie punktu końcowego
        ax.scatter(x[-1], v[-1], color=color, s=50, marker='s', zorder=5)

    ax.set_xlabel('Położenie x [m]', fontsize=12)
    ax.set_ylabel('Prędkość v [m/s]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_prediction_comparison(
    time: np.ndarray,
    target: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    figsize: Tuple[int, int] = (14, 6),
    title: str = 'Porównanie predykcji',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Tworzy wykres porównujący predykcje dla obu cech (x i v).

    Args:
        time: Wektor czasu
        target: Wartości docelowe (T, 2)
        mu: Predykowana średnia (T, 2)
        sigma: Predykowane odchylenie std (T, 2)
        figsize: Rozmiar figury
        title: Tytuł wykresu
        save_path: Opcjonalna ścieżka do zapisu

    Returns:
        Obiekt Figure matplotlib
    """
    # Konwersja do numpy
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    feature_names = ['Położenie x [m]', 'Prędkość v [m/s]']
    feature_labels = ['x', 'v']

    for idx, (ax, name, label) in enumerate(zip(axes, feature_names, feature_labels)):
        # Przedziały ufności
        ax.fill_between(
            time,
            mu[:, idx] - 2 * sigma[:, idx],
            mu[:, idx] + 2 * sigma[:, idx],
            alpha=0.2, color='blue', label='±2σ'
        )
        ax.fill_between(
            time,
            mu[:, idx] - sigma[:, idx],
            mu[:, idx] + sigma[:, idx],
            alpha=0.3, color='blue', label='±1σ'
        )

        # Wartości rzeczywiste i predykcja
        ax.plot(time, target[:, idx], 'g-', linewidth=2, label='Rzeczywiste')
        ax.plot(time, mu[:, idx], 'r--', linewidth=2, label='Predykcja')

        ax.set_xlabel('Czas [s]', fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f'Predykcja {label}', fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_uncertainty_evolution(
    time: np.ndarray,
    sigma: np.ndarray,
    figsize: Tuple[int, int] = (10, 5),
    title: str = 'Ewolucja niepewności',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Tworzy wykres pokazujący jak niepewność rośnie w czasie.

    Args:
        time: Wektor czasu
        sigma: Predykowane odchylenia std (T, features)
        figsize: Rozmiar figury
        title: Tytuł wykresu
        save_path: Opcjonalna ścieżka do zapisu

    Returns:
        Obiekt Figure matplotlib
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    feature_names = ['σ(x)', 'σ(v)']
    colors = ['blue', 'red']

    for idx, (name, color) in enumerate(zip(feature_names, colors)):
        ax.plot(time, sigma[:, idx], color=color, linewidth=2, label=name)

    ax.set_xlabel('Czas [s]', fontsize=12)
    ax.set_ylabel('Odchylenie standardowe σ', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_full_trajectory_with_prediction(
    full_trajectory: np.ndarray,
    full_time: np.ndarray,
    input_start_idx: int,
    T_in: int,
    T_out: int,
    mu_seq: np.ndarray,
    sigma_seq: np.ndarray,
    feature_idx: int = 0,
    feature_name: str = 'Położenie x [m]',
    figsize: Tuple[int, int] = (16, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Tworzy wykres pełnej trajektorii z zaznaczoną predykcją.

    Pokazuje całą oryginalną trajektorię, z wyróżnieniem:
    - Okna wejściowego (input)
    - Fragmentu docelowego (target)
    - Predykcji z przedziałami ufności

    Args:
        full_trajectory: Pełna trajektoria (num_steps, features)
        full_time: Pełny wektor czasu
        input_start_idx: Indeks początkowy okna wejściowego
        T_in: Długość okna wejściowego
        T_out: Długość horyzontu predykcji
        mu_seq: Predykowana średnia (T_out, features)
        sigma_seq: Predykowane odchylenie std (T_out, features)
        feature_idx: Indeks cechy do wizualizacji (0 dla x, 1 dla v)
        feature_name: Nazwa cechy do etykiet
        figsize: Rozmiar figury
        title: Opcjonalny tytuł wykresu
        save_path: Opcjonalna ścieżka do zapisu wykresu

    Returns:
        Obiekt Figure matplotlib
    """
    # Konwersja tensorów do numpy
    if isinstance(full_trajectory, torch.Tensor):
        full_trajectory = full_trajectory.detach().cpu().numpy()
    if isinstance(mu_seq, torch.Tensor):
        mu_seq = mu_seq.detach().cpu().numpy()
    if isinstance(sigma_seq, torch.Tensor):
        sigma_seq = sigma_seq.detach().cpu().numpy()

    # Indeksy
    input_end_idx = input_start_idx + T_in
    output_end_idx = input_end_idx + T_out

    # Wyciągnięcie danych
    full_values = full_trajectory[:, feature_idx]
    mu_values = mu_seq[:, feature_idx]
    sigma_values = sigma_seq[:, feature_idx]

    # Czasy dla predykcji
    time_prediction = full_time[input_end_idx:output_end_idx]

    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=figsize)

    # 1. Pełna trajektoria (cała, szara, w tle)
    ax.plot(full_time, full_values, 'gray', linewidth=1.5, alpha=0.5, label='Pełna trajektoria')

    # 2. Okno wejściowe (niebieskie, grubsze)
    ax.plot(full_time[input_start_idx:input_end_idx],
            full_values[input_start_idx:input_end_idx],
            'b-', linewidth=2.5, label=f'Okno wejściowe (T_in={T_in})')

    # 3. Fragment docelowy (zielony)
    ax.plot(full_time[input_end_idx:output_end_idx],
            full_values[input_end_idx:output_end_idx],
            'g-', linewidth=2.5, label=f'Wartości rzeczywiste (T_out={T_out})')

    # 4. Przedziały ufności
    ax.fill_between(
        time_prediction,
        mu_values - 2 * sigma_values,
        mu_values + 2 * sigma_values,
        alpha=0.2, color='red', label='±2σ (95.45% CI)'
    )
    ax.fill_between(
        time_prediction,
        mu_values - sigma_values,
        mu_values + sigma_values,
        alpha=0.3, color='red', label='±1σ (68.27% CI)'
    )

    # 5. Predykcja (czerwona, przerywana)
    ax.plot(time_prediction, mu_values, 'r--', linewidth=2.5, label='Predykcja (μ)')

    # 6. Linie pionowe oznaczające granice
    ax.axvline(x=full_time[input_start_idx], color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=full_time[input_end_idx], color='orange', linestyle='--', linewidth=2, alpha=0.8,
               label='Granica input/output')
    if output_end_idx < len(full_time):
        ax.axvline(x=full_time[output_end_idx], color='green', linestyle=':', linewidth=1.5, alpha=0.7)

    # 7. Zaznaczenie obszaru predykcji (tło)
    ax.axvspan(full_time[input_end_idx], full_time[min(output_end_idx, len(full_time)-1)],
               alpha=0.1, color='yellow', label='Horyzont predykcji')

    # Formatowanie
    ax.set_xlabel('Czas [s]', fontsize=12)
    ax.set_ylabel(feature_name, fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Pełna trajektoria z predykcją - {feature_name}', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_recursive_prediction(
    full_trajectory: np.ndarray,
    full_time: np.ndarray,
    initial_input: np.ndarray,
    recursive_predictions: List[Dict],
    T_in: int,
    T_out: int,
    feature_idx: int = 0,
    feature_name: str = 'Położenie x [m]',
    figsize: Tuple[int, int] = (18, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    oscillator_params: Optional[Dict] = None
) -> plt.Figure:
    """
    Tworzy wykres rekurencyjnej predykcji.

    Pokazuje jak model przewiduje trajektorię krok po kroku,
    używając własnych predykcji jako wejścia dla kolejnych kroków.

    Args:
        full_trajectory: Pełna trajektoria rzeczywista (num_steps, features)
        full_time: Pełny wektor czasu
        initial_input: Początkowe okno wejściowe (T_in, features)
        recursive_predictions: Lista predykcji rekurencyjnych, każda zawiera:
            - 'mu': Predykowana średnia (T_out, features)
            - 'sigma': Predykowane odchylenie std (T_out, features)
            - 'time_start': Czas początkowy predykcji
        T_in: Długość okna wejściowego
        T_out: Długość horyzontu predykcji
        feature_idx: Indeks cechy do wizualizacji
        feature_name: Nazwa cechy
        figsize: Rozmiar figury
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu
        oscillator_params: Parametry oscylatora

    Returns:
        Obiekt Figure matplotlib
    """
    if isinstance(full_trajectory, torch.Tensor):
        full_trajectory = full_trajectory.detach().cpu().numpy()
    if isinstance(initial_input, torch.Tensor):
        initial_input = initial_input.detach().cpu().numpy()

    # Kolory dla kolejnych predykcji (gradient od zielonego do czerwonego)
    n_preds = len(recursive_predictions)
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n_preds))

    fig, ax = plt.subplots(figsize=figsize)

    # 1. Pełna trajektoria rzeczywista (szara, w tle)
    full_values = full_trajectory[:, feature_idx]
    ax.plot(full_time, full_values, 'gray', linewidth=2, alpha=0.5,
            label='Trajektoria rzeczywista')

    # 2. Początkowe okno wejściowe (niebieskie)
    input_time = full_time[:T_in]
    input_values = initial_input[:, feature_idx]
    ax.plot(input_time, input_values, 'b-', linewidth=3, alpha=0.9,
            label=f'Wejście początkowe (T_in={T_in})')

    # 3. Predykcje rekurencyjne
    current_time = full_time[T_in - 1]  # Ostatni punkt wejścia

    for i, pred in enumerate(recursive_predictions):
        mu = pred['mu']
        sigma = pred['sigma']

        if isinstance(mu, torch.Tensor):
            mu = mu.detach().cpu().numpy()
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.detach().cpu().numpy()

        mu_values = mu[:, feature_idx]
        sigma_values = sigma[:, feature_idx]

        # Czas dla tej predykcji
        dt = full_time[1] - full_time[0]
        pred_time = np.arange(T_out) * dt + current_time + dt

        # Przedziały ufności
        ax.fill_between(
            pred_time,
            mu_values - 2 * sigma_values,
            mu_values + 2 * sigma_values,
            alpha=0.15, color=colors[i]
        )
        ax.fill_between(
            pred_time,
            mu_values - sigma_values,
            mu_values + sigma_values,
            alpha=0.25, color=colors[i]
        )

        # Predykcja
        label = f'Predykcja #{i+1}' if i < 3 or i == n_preds - 1 else None
        ax.plot(pred_time, mu_values, '--', color=colors[i], linewidth=2.5,
                alpha=0.9, label=label)

        # Linia graniczna
        ax.axvline(x=current_time + dt, color=colors[i], linestyle=':',
                   linewidth=1, alpha=0.5)

        # Aktualizacja czasu dla następnej predykcji
        current_time = pred_time[-1]

    # Oznaczenie końca predykcji
    total_pred_time = T_in * dt + n_preds * T_out * dt
    ax.axvline(x=total_pred_time, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Koniec predykcji ({n_preds} kroków)')

    # Formatowanie
    ax.set_xlabel('Czas [s]', fontsize=12)
    ax.set_ylabel(feature_name, fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Tytuł
    if title:
        full_title = title
    else:
        full_title = f'Predykcja rekurencyjna ({n_preds} kroków)'

    if oscillator_params:
        param_str = f"m={oscillator_params.get('mass', '?')}, " \
                    f"c={oscillator_params.get('damping', '?')}, " \
                    f"k={oscillator_params.get('stiffness', '?')}"
        full_title += f'\n({param_str})'

    ax.set_title(full_title, fontsize=13)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_multi_prediction_trajectory(
    full_trajectory: np.ndarray,
    full_time: np.ndarray,
    predictions: List[Dict],
    T_in: int,
    T_out: int,
    feature_idx: int = 0,
    feature_name: str = 'Położenie x [m]',
    figsize: Tuple[int, int] = (16, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    oscillator_params: Optional[Dict] = None
) -> plt.Figure:
    """
    Tworzy wykres pełnej trajektorii z wieloma predykcjami w różnych punktach.

    Args:
        full_trajectory: Pełna trajektoria (num_steps, features)
        full_time: Pełny wektor czasu
        predictions: Lista słowników z predykcjami, każdy zawiera:
            - 'start_idx': Indeks początkowy okna wejściowego
            - 'mu': Predykowana średnia (T_out, features)
            - 'sigma': Predykowane odchylenie std (T_out, features)
        T_in: Długość okna wejściowego
        T_out: Długość horyzontu predykcji
        feature_idx: Indeks cechy do wizualizacji (0 dla x, 1 dla v)
        feature_name: Nazwa cechy do etykiet
        figsize: Rozmiar figury
        title: Opcjonalny tytuł wykresu
        save_path: Opcjonalna ścieżka do zapisu wykresu
        oscillator_params: Opcjonalne parametry oscylatora do wyświetlenia

    Returns:
        Obiekt Figure matplotlib
    """
    # Konwersja tensorów do numpy
    if isinstance(full_trajectory, torch.Tensor):
        full_trajectory = full_trajectory.detach().cpu().numpy()

    # Kolory dla różnych predykcji
    colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))

    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=figsize)

    # 1. Pełna trajektoria (cała, szara, w tle)
    full_values = full_trajectory[:, feature_idx]
    ax.plot(full_time, full_values, 'gray', linewidth=2, alpha=0.6, label='Pełna trajektoria')

    # 2. Predykcje w różnych punktach
    for i, pred in enumerate(predictions):
        start_idx = pred['start_idx']
        mu = pred['mu']
        sigma = pred['sigma']

        if isinstance(mu, torch.Tensor):
            mu = mu.detach().cpu().numpy()
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.detach().cpu().numpy()

        input_end_idx = start_idx + T_in
        output_end_idx = input_end_idx + T_out

        mu_values = mu[:, feature_idx]
        sigma_values = sigma[:, feature_idx]

        # Czas dla predykcji
        time_prediction = full_time[input_end_idx:output_end_idx]

        # Okno wejściowe
        ax.plot(full_time[start_idx:input_end_idx],
                full_values[start_idx:input_end_idx],
                color=colors[i], linewidth=2.5, alpha=0.8,
                label=f'Input #{i+1} (t={full_time[start_idx]:.1f}s)')

        # Wartości rzeczywiste (target)
        ax.plot(full_time[input_end_idx:output_end_idx],
                full_values[input_end_idx:output_end_idx],
                color=colors[i], linewidth=2, linestyle='-', alpha=0.5)

        # Przedziały ufności
        ax.fill_between(
            time_prediction,
            mu_values - 2 * sigma_values,
            mu_values + 2 * sigma_values,
            alpha=0.15, color=colors[i]
        )
        ax.fill_between(
            time_prediction,
            mu_values - sigma_values,
            mu_values + sigma_values,
            alpha=0.25, color=colors[i]
        )

        # Predykcja
        ax.plot(time_prediction, mu_values, '--', color=colors[i],
                linewidth=2.5, alpha=0.9, label=f'Predykcja #{i+1}')

        # Linia graniczna input/output
        ax.axvline(x=full_time[input_end_idx], color=colors[i],
                   linestyle=':', linewidth=1.5, alpha=0.5)

    # Formatowanie
    ax.set_xlabel('Czas [s]', fontsize=12)
    ax.set_ylabel(feature_name, fontsize=12)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Tytuł z parametrami oscylatora
    if title:
        full_title = title
    else:
        full_title = f'Trajektoria z {len(predictions)} predykcjami'

    if oscillator_params:
        param_str = f"m={oscillator_params.get('mass', '?')}, " \
                    f"c={oscillator_params.get('damping', '?')}, " \
                    f"k={oscillator_params.get('stiffness', '?')}, " \
                    f"x₀={oscillator_params.get('x0', '?')}, " \
                    f"v₀={oscillator_params.get('v0', '?')}"
        full_title += f'\n({param_str})'

    ax.set_title(full_title, fontsize=13)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_calibration_diagram(
    expected_coverage: List[float],
    actual_coverage: List[float],
    figsize: Tuple[int, int] = (6, 6),
    title: str = 'Diagram kalibracji',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Tworzy diagram kalibracji modelu probabilistycznego.

    Idealnie skalibrowany model powinien leżeć na przekątnej.

    Args:
        expected_coverage: Oczekiwane prawdopodobieństwa pokrycia
        actual_coverage: Rzeczywiste prawdopodobieństwa pokrycia
        figsize: Rozmiar figury
        title: Tytuł wykresu
        save_path: Opcjonalna ścieżka do zapisu

    Returns:
        Obiekt Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Przekątna (idealna kalibracja)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Idealna kalibracja')

    # Rzeczywista kalibracja
    ax.plot(
        expected_coverage, actual_coverage,
        'bo-', linewidth=2, markersize=8,
        label='Model'
    )

    # Wypełnienie obszaru błędu
    ax.fill_between(
        expected_coverage, expected_coverage, actual_coverage,
        alpha=0.2, color='red'
    )

    ax.set_xlabel('Oczekiwane pokrycie', fontsize=12)
    ax.set_ylabel('Rzeczywiste pokrycie', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Test modułu wizualizacji
    print("Test wizualizacji")
    print("=" * 60)

    # Generowanie przykładowych danych
    np.random.seed(42)

    T_in = 50
    T_out = 30
    dt = 0.01

    time_input = np.arange(0, T_in * dt, dt)
    time_output = np.arange(T_in * dt, (T_in + T_out) * dt, dt)

    # Symulacja oscylatora tłumionego
    omega = 2.0
    zeta = 0.1
    t_full = np.arange(0, (T_in + T_out) * dt, dt)

    x = np.exp(-zeta * omega * t_full) * np.cos(omega * np.sqrt(1 - zeta**2) * t_full)
    v = -omega * np.exp(-zeta * omega * t_full) * (
        zeta * np.cos(omega * np.sqrt(1 - zeta**2) * t_full) +
        np.sqrt(1 - zeta**2) * np.sin(omega * np.sqrt(1 - zeta**2) * t_full)
    )

    full_seq = np.column_stack([x, v])
    input_seq = full_seq[:T_in]
    target_seq = full_seq[T_in:T_in + T_out]

    # Symulacja predykcji z szumem
    mu_seq = target_seq + np.random.randn(T_out, 2) * 0.05
    sigma_seq = np.abs(np.linspace(0.02, 0.15, T_out)[:, np.newaxis] * np.ones((T_out, 2)))

    # Test plot_prediction_with_uncertainty
    print("\n1. Test plot_prediction_with_uncertainty:")
    fig1 = plot_prediction_with_uncertainty(
        time_input=time_input,
        time_output=time_output,
        input_seq=input_seq,
        target_seq=target_seq,
        mu_seq=mu_seq,
        sigma_seq=sigma_seq,
        feature_idx=0,
        feature_name='Położenie x [m]',
        save_path='data/synthetic/test_prediction.png'
    )
    print("   Zapisano: data/synthetic/test_prediction.png")
    plt.close(fig1)

    # Test plot_training_curves
    print("\n2. Test plot_training_curves:")
    epochs = 50
    train_losses = [2.0 - 1.5 * (1 - np.exp(-i/10)) + np.random.rand() * 0.1 for i in range(epochs)]
    val_losses = [2.0 - 1.4 * (1 - np.exp(-i/10)) + np.random.rand() * 0.15 for i in range(epochs)]

    fig2 = plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        save_path='data/synthetic/test_training_curves.png'
    )
    print("   Zapisano: data/synthetic/test_training_curves.png")
    plt.close(fig2)

    # Test plot_phase_space
    print("\n3. Test plot_phase_space:")
    trajectories = [input_seq, target_seq, mu_seq]
    labels = ['Wejście', 'Cel', 'Predykcja']

    fig3 = plot_phase_space(
        trajectories=trajectories,
        labels=labels,
        save_path='data/synthetic/test_phase_space.png'
    )
    print("   Zapisano: data/synthetic/test_phase_space.png")
    plt.close(fig3)

    # Test plot_prediction_comparison
    print("\n4. Test plot_prediction_comparison:")
    fig4 = plot_prediction_comparison(
        time=time_output,
        target=target_seq,
        mu=mu_seq,
        sigma=sigma_seq,
        save_path='data/synthetic/test_comparison.png'
    )
    print("   Zapisano: data/synthetic/test_comparison.png")
    plt.close(fig4)

    # Test plot_uncertainty_evolution
    print("\n5. Test plot_uncertainty_evolution:")
    fig5 = plot_uncertainty_evolution(
        time=time_output,
        sigma=sigma_seq,
        save_path='data/synthetic/test_uncertainty.png'
    )
    print("   Zapisano: data/synthetic/test_uncertainty.png")
    plt.close(fig5)

    # Test plot_calibration_diagram
    print("\n6. Test plot_calibration_diagram:")
    expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    actual = [0.12, 0.22, 0.28, 0.38, 0.52, 0.58, 0.72, 0.78, 0.88]

    fig6 = plot_calibration_diagram(
        expected_coverage=expected,
        actual_coverage=actual,
        save_path='data/synthetic/test_calibration.png'
    )
    print("   Zapisano: data/synthetic/test_calibration.png")
    plt.close(fig6)

    print("\n" + "=" * 60)
    print("Test wizualizacji zakończony pomyślnie!")
