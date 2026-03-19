"""
Moduł generacji syntetycznych danych czasowych.

Implementuje oscylator harmoniczny tłumiony jako źródło danych
testowych do walidacji modelu predykcji szeregów czasowych.

Równanie ruchu oscylatora:
    m * x''(t) + c * x'(t) + k * x(t) = 0

gdzie:
    m - masa [kg]
    c - współczynnik tłumienia [Ns/m]
    k - sztywność [N/m]
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import odeint


@dataclass
class DimensionlessParams:
    """
    Parametry bezwymiarowe oscylatora.

    Równanie bezwymiarowe: d²x/dτ² + 2ζ(dx/dτ) + x = 0
    gdzie τ = ω₀·t jest czasem bezwymiarowym.

    Attributes:
        zeta: Współczynnik tłumienia bezwymiarowy ζ ∈ [0, 1)
    """
    zeta: float

    @property
    def is_underdamped(self) -> bool:
        """Czy układ jest podkrytycznie tłumiony (oscylacje gasnące)."""
        return self.zeta < 1.0

    @property
    def omega_d_normalized(self) -> float:
        """Znormalizowana częstość tłumiona (w czasie bezwymiarowym)."""
        if self.zeta >= 1.0:
            return 0.0
        return np.sqrt(1 - self.zeta**2)


@dataclass
class ForcedDimensionlessParams:
    """
    Parametry bezwymiarowe oscylatora wymuszonego.

    Równanie bezwymiarowe: d²x/dτ² + 2ζ(dx/dτ) + x = F₀·cos(Ωτ)
    gdzie τ = ω₀·t jest czasem bezwymiarowym,
    Ω = ω/ω₀ jest bezwymiarową częstością wymuszenia.

    Attributes:
        zeta: Współczynnik tłumienia bezwymiarowy ζ ≥ 0
        omega: Bezwymiarowa częstość wymuszenia Ω > 0
        f0: Bezwymiarowa amplituda wymuszenia F₀
    """
    zeta: float
    omega: float
    f0: float = 1.0

    @property
    def is_underdamped(self) -> bool:
        """Czy układ jest podkrytycznie tłumiony."""
        return self.zeta < 1.0

    @property
    def is_near_resonance(self) -> bool:
        """Czy częstość wymuszenia jest blisko rezonansu (Ω ≈ 1)."""
        return 0.7 <= self.omega <= 1.3


@dataclass
class OscillatorParams:
    """
    Parametry fizyczne oscylatora tłumionego.

    Attributes:
        mass: Masa układu [kg]
        damping: Współczynnik tłumienia [Ns/m]
        stiffness: Sztywność sprężyny [N/m]
    """
    mass: float
    damping: float
    stiffness: float

    @property
    def omega_0(self) -> float:
        """Częstość własna nietłumiona [rad/s]."""
        return np.sqrt(self.stiffness / self.mass)

    @property
    def zeta(self) -> float:
        """Współczynnik tłumienia bezwymiarowy."""
        return self.damping / (2 * np.sqrt(self.stiffness * self.mass))

    @property
    def omega_d(self) -> float:
        """Częstość własna tłumiona [rad/s]."""
        if self.zeta >= 1.0:
            return 0.0
        return self.omega_0 * np.sqrt(1 - self.zeta**2)


class DampedOscillator:
    """
    Klasa reprezentująca oscylator harmoniczny tłumiony.

    Generuje trajektorie ruchu na podstawie równania różniczkowego
    drugiego rzędu, rozwiązywanego numerycznie metodą odeint.

    Attributes:
        params: Parametry fizyczne oscylatora
    """

    def __init__(
        self,
        mass: float = 1.0,
        damping: float = 0.2,
        stiffness: float = 1.0
    ):
        """
        Inicjalizacja oscylatora.

        Args:
            mass: Masa układu [kg]
            damping: Współczynnik tłumienia [Ns/m]
            stiffness: Sztywność sprężyny [N/m]
        """
        self.params = OscillatorParams(
            mass=mass,
            damping=damping,
            stiffness=stiffness
        )

    def _equations_of_motion(
        self,
        state: np.ndarray,
        t: float
    ) -> List[float]:
        """
        Równania ruchu w postaci układu równań pierwszego rzędu.

        Przekształcenie:
            x' = v
            v' = -(c/m)*v - (k/m)*x

        Args:
            state: Wektor stanu [x, v]
            t: Czas (nieużywany, wymagany przez odeint)

        Returns:
            Lista pochodnych [dx/dt, dv/dt]
        """
        x, v = state
        dxdt = v
        dvdt = (
            -self.params.damping / self.params.mass * v
            - self.params.stiffness / self.params.mass * x
        )
        return [dxdt, dvdt]

    def generate_trajectory(
        self,
        x0: float,
        v0: float,
        t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generuje trajektorię ruchu dla zadanych warunków początkowych.

        Args:
            x0: Położenie początkowe [m]
            v0: Prędkość początkowa [m/s]
            t: Wektor czasu [s]

        Returns:
            Tuple zawierający:
                - x: Wektor położeń [m]
                - v: Wektor prędkości [m/s]
        """
        initial_state = [x0, v0]
        solution = odeint(self._equations_of_motion, initial_state, t)

        x = solution[:, 0]
        v = solution[:, 1]

        return x, v

    def generate_state_space(
        self,
        x0: float,
        v0: float,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Generuje trajektorię w przestrzeni stanów [x, v].

        Args:
            x0: Położenie początkowe [m]
            v0: Prędkość początkowa [m/s]
            t: Wektor czasu [s]

        Returns:
            Macierz stanów o wymiarach (len(t), 2)
        """
        x, v = self.generate_trajectory(x0, v0, t)
        return np.column_stack([x, v])


class SimpleHarmonicOscillator(DampedOscillator):
    """
    Oscylator harmoniczny prosty (bez tłumienia).

    Równanie ruchu:
        m * x''(t) + k * x(t) = 0

    Jest to szczególny przypadek oscylatora tłumionego z c=0.
    """

    def __init__(
        self,
        mass: float = 1.0,
        stiffness: float = 1.0
    ):
        """
        Inicjalizacja oscylatora prostego.

        Args:
            mass: Masa układu [kg]
            stiffness: Sztywność sprężyny [N/m]
        """
        super().__init__(mass=mass, damping=0.0, stiffness=stiffness)


class DimensionlessOscillator:
    """
    Oscylator w parametryzacji bezwymiarowej.

    Równanie ruchu w czasie bezwymiarowym τ = ω₀·t:
        d²x/dτ² + 2ζ(dx/dτ) + x = 0

    Stałe warunki początkowe: x(0)=1, dx/dτ(0)=0

    Korzyści:
    - Dla stałych warunków początkowych dynamika zależy TYLKO od ζ
    - Model uczy się uniwersalnych zależności w przestrzeni bezwymiarowej
    - Lepsza generalizacja na różne układy fizyczne

    Attributes:
        params: Parametry bezwymiarowe oscylatora (ζ)
    """

    def __init__(
        self,
        zeta: float = 0.1
    ):
        """
        Inicjalizacja oscylatora bezwymiarowego.

        Args:
            zeta: Współczynnik tłumienia bezwymiarowy ζ ∈ [0, 1)
        """
        if zeta < 0:
            raise ValueError("Współczynnik tłumienia ζ musi być nieujemny")

        self.params = DimensionlessParams(zeta=zeta)

    def _equations_of_motion(
        self,
        state: np.ndarray,
        tau: float
    ) -> List[float]:
        """
        Równania ruchu w postaci bezwymiarowej.

        Przekształcenie:
            dx/dτ = v_τ
            dv_τ/dτ = -2ζ·v_τ - x

        Args:
            state: Wektor stanu [x, v_τ] gdzie v_τ = dx/dτ
            tau: Czas bezwymiarowy (nieużywany, wymagany przez odeint)

        Returns:
            Lista pochodnych [dx/dτ, dv_τ/dτ]
        """
        x, v_tau = state
        dxdtau = v_tau
        dvdtau = -2 * self.params.zeta * v_tau - x
        return [dxdtau, dvdtau]

    def generate_trajectory(
        self,
        tau: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generuje trajektorię w czasie bezwymiarowym.

        Warunki początkowe są stałe: x(0)=1, dx/dτ(0)=0
        Dzięki temu dynamika zależy tylko od ζ.

        Args:
            tau: Wektor czasu bezwymiarowego τ = ω₀·t

        Returns:
            Tuple zawierający:
                - x: Wektor położeń (bezwymiarowych)
                - v_tau: Wektor prędkości bezwymiarowych (dx/dτ)
        """
        # Stałe warunki początkowe - kluczowa cecha parametryzacji bezwymiarowej
        initial_state = [1.0, 0.0]  # x₀=1, v₀=0

        solution = odeint(self._equations_of_motion, initial_state, tau)

        x = solution[:, 0]
        v_tau = solution[:, 1]

        return x, v_tau

    def generate_state_space(
        self,
        tau: np.ndarray
    ) -> np.ndarray:
        """
        Generuje trajektorię w przestrzeni stanów [x, dx/dτ].

        Args:
            tau: Wektor czasu bezwymiarowego

        Returns:
            Macierz stanów o wymiarach (len(tau), 2)
        """
        x, v_tau = self.generate_trajectory(tau)
        return np.column_stack([x, v_tau])


# Stałe definiujące reżimy oscylatora wymuszonego bezwymiarowego
FORCED_ZETA_GROUPS = {
    'Z': (0.00, 0.007),  # praktycznie bez tłumienia
    'A': (0.007, 0.05),  # minimalne tłumienie
    'B': (0.05, 0.25),   # lekkie tłumienie
    'C': (0.25, 0.85),   # spore tłumienie
}
FORCED_OMEGA_GROUPS = {
    '1': (0.3, 0.9),    # poniżej rezonansu
    '2': (0.9, 1.1),    # okolice rezonansu (zawężone)
    '3': (1.1, 2.5),    # powyżej rezonansu
}
FORCED_F0 = 1.0


class ForcedDimensionlessOscillator:
    """
    Oscylator wymuszony w parametryzacji bezwymiarowej.

    Równanie ruchu w czasie bezwymiarowym τ = ω₀·t:
        d²x/dτ² + 2ζ(dx/dτ) + x = F₀·cos(Ωτ)

    gdzie Ω = ω/ω₀ jest bezwymiarową częstością wymuszenia.

    Stałe warunki początkowe: x(0)=1, dx/dτ(0)=0

    Attributes:
        params: Parametry bezwymiarowe oscylatora (ζ, Ω, F₀)
    """

    def __init__(
        self,
        zeta: float = 0.1,
        omega: float = 1.0,
        f0: float = 1.0
    ):
        """
        Inicjalizacja oscylatora wymuszonego bezwymiarowego.

        Args:
            zeta: Współczynnik tłumienia bezwymiarowy ζ ≥ 0
            omega: Bezwymiarowa częstość wymuszenia Ω > 0
            f0: Bezwymiarowa amplituda wymuszenia F₀
        """
        if zeta < 0:
            raise ValueError("Współczynnik tłumienia ζ musi być nieujemny")
        if omega <= 0:
            raise ValueError("Częstość wymuszenia Ω musi być dodatnia")

        self.params = ForcedDimensionlessParams(zeta=zeta, omega=omega, f0=f0)

    def __init_onset__(self, tau_onset: float = 0.0):
        """Ustaw opóźniony start wymuszenia — przed τ_onset układ w spoczynku."""
        self._tau_onset = tau_onset

    def _equations_of_motion(
        self,
        state: np.ndarray,
        tau: float
    ) -> List[float]:
        """
        Równania ruchu oscylatora wymuszonego w postaci bezwymiarowej.

        Przekształcenie:
            dx/dτ = v_τ
            dv_τ/dτ = -2ζ·v_τ - x + F₀·cos(Ωτ)  (dla τ ≥ τ_onset)

        Args:
            state: Wektor stanu [x, v_τ] gdzie v_τ = dx/dτ
            tau: Czas bezwymiarowy

        Returns:
            Lista pochodnych [dx/dτ, dv_τ/dτ]
        """
        x, v_tau = state
        dxdtau = v_tau
        # Wymuszenie aktywne dopiero od τ_onset
        tau_onset = getattr(self, '_tau_onset', 0.0)
        forcing = self.params.f0 * np.cos(self.params.omega * tau) if tau >= tau_onset else 0.0
        dvdtau = (
            -2 * self.params.zeta * v_tau
            - x
            + forcing
        )
        return [dxdtau, dvdtau]

    def generate_trajectory(
        self,
        tau: np.ndarray,
        tau_onset: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generuje trajektorię w czasie bezwymiarowym.

        Warunki początkowe są stałe: x(0)=1, dx/dτ(0)=0
        Opcjonalnie z opóźnionym startem wymuszenia (τ_onset).

        Args:
            tau: Wektor czasu bezwymiarowego τ = ω₀·t
            tau_onset: Moment startu wymuszenia (None = natychmiast)

        Returns:
            Tuple zawierający:
                - x: Wektor położeń (bezwymiarowych)
                - v_tau: Wektor prędkości bezwymiarowych (dx/dτ)
        """
        if tau_onset is not None:
            self._tau_onset = tau_onset
            initial_state = [0.0, 0.0]  # układ startuje ze spoczynku
        else:
            self._tau_onset = 0.0
            initial_state = [1.0, 0.0]  # x₀=1, v₀=0

        solution = odeint(self._equations_of_motion, initial_state, tau)

        x = solution[:, 0]
        v_tau = solution[:, 1]

        return x, v_tau

    def generate_state_space(
        self,
        tau: np.ndarray
    ) -> np.ndarray:
        """
        Generuje trajektorię w przestrzeni stanów [x, dx/dτ].

        Args:
            tau: Wektor czasu bezwymiarowego

        Returns:
            Macierz stanów o wymiarach (len(tau), 2)
        """
        x, v_tau = self.generate_trajectory(tau)
        return np.column_stack([x, v_tau])


def add_noise(
    data: np.ndarray,
    noise_std: float = 0.01,
    seed: Optional[int] = None,
    relative: bool = True
) -> np.ndarray:
    """
    Dodaje szum gaussowski do danych.

    Args:
        data: Dane wejściowe (dowolny kształt)
        noise_std: Odchylenie standardowe szumu.
            Jeśli relative=True: szum proporcjonalny do wartości (noise_std jako ułamek, np. 0.03 = 3%)
            Jeśli relative=False: szum addytywny stały
        seed: Ziarno generatora losowego (opcjonalne)
        relative: Jeśli True, szum proporcjonalny do |wartości| (symulacja czujnika)

    Returns:
        Dane z dodanym szumem
    """
    if seed is not None:
        np.random.seed(seed)

    if relative:
        # Szum proporcjonalny: σ = noise_std * |wartość|
        # Minimum σ = noise_std * min_floor (szum bazowy czujnika niezależny od wartości)
        # Dla noise_std=0.03: min σ = 0.03 * 0.3 = 0.009 ≈ 1% amplitudy
        min_floor = 0.3
        sigma = noise_std * np.maximum(np.abs(data), min_floor)
        noise = np.random.normal(0, 1, data.shape) * sigma
    else:
        noise = np.random.normal(0, noise_std, data.shape)
    return data + noise


def add_dropout(data: np.ndarray, dropout_rate: float = 0.02) -> np.ndarray:
    """
    Symulacja brakujących próbek (utrata pakietów UDP, luki w transmisji).
    Zastępuje losowe próbki interpolacją z sąsiednich wartości.

    Args:
        data: Macierz stanów (steps, 2)
        dropout_rate: Odsetek próbek do usunięcia (0.0–0.1)
    """
    result = data.copy()
    n = len(data)
    num_drop = max(1, int(n * dropout_rate))
    # Losowe indeksy do usunięcia (nie na brzegach)
    drop_idx = np.random.choice(range(1, n - 1), size=min(num_drop, n - 2), replace=False)
    for idx in drop_idx:
        # Zastąp interpolacją liniową z sąsiadów
        result[idx] = (result[idx - 1] + result[idx + 1]) / 2.0
    return result


def add_spikes(data: np.ndarray, spike_rate: float = 0.005, spike_magnitude: float = 5.0) -> np.ndarray:
    """
    Symulacja nagłych skoków wartości (zakłócenia EM, błędy komunikacji).

    Args:
        data: Macierz stanów (steps, 2)
        spike_rate: Odsetek próbek ze skokiem (0.0–0.02)
        spike_magnitude: Mnożnik amplitudy skoku (relative to data std)
    """
    result = data.copy()
    n = len(data)
    num_spikes = max(0, int(n * spike_rate))
    if num_spikes == 0:
        return result
    spike_idx = np.random.choice(n, size=num_spikes, replace=False)
    for col in range(data.shape[1]):
        col_std = np.std(data[:, col])
        if col_std > 0:
            result[spike_idx, col] += np.random.choice([-1, 1], size=num_spikes) * spike_magnitude * col_std
    return result


def add_drift(data: np.ndarray, drift_magnitude: float = 0.1) -> np.ndarray:
    """
    Symulacja wolnozmiennego dryftu offsetu (temperatura, mechanika).
    Dodaje losową wolnozmienną składową do sygnału.

    Args:
        data: Macierz stanów (steps, 2)
        drift_magnitude: Amplituda dryftu (względem std danych)
    """
    result = data.copy()
    n = len(data)
    # Losowy dryft: sinusoida o bardzo niskiej częstotliwości
    num_waves = np.random.uniform(0.5, 2.0)
    phase = np.random.uniform(0, 2 * np.pi)
    drift_wave = np.sin(np.linspace(0, num_waves * 2 * np.pi, n) + phase)
    for col in range(data.shape[1]):
        col_std = np.std(data[:, col])
        if col_std > 0:
            result[:, col] += drift_magnitude * col_std * drift_wave
    return result


def add_saturation(data: np.ndarray, clip_percentile: float = 95) -> np.ndarray:
    """
    Symulacja saturacji czujnika (ograniczony zakres pomiarowy).
    Obcina wartości powyżej/poniżej losowo wybranego percentyla.

    Args:
        data: Macierz stanów (steps, 2)
        clip_percentile: Percentyl obcięcia (80–99)
    """
    result = data.copy()
    for col in range(data.shape[1]):
        upper = np.percentile(data[:, col], clip_percentile)
        lower = np.percentile(data[:, col], 100 - clip_percentile)
        result[:, col] = np.clip(result[:, col], lower, upper)
    return result


def apply_sensor_augmentation(
    state: np.ndarray,
    noise_std: float = 0.03,
    dropout_rate: float = 0.02,
    spike_rate: float = 0.005,
    drift_magnitude: float = 0.1,
    saturation_clip: float = 95,
    enable_dropout: bool = True,
    enable_spikes: bool = True,
    enable_drift: bool = True,
    enable_saturation: bool = True,
) -> np.ndarray:
    """
    Łączna augmentacja danych symulująca realistyczne warunki pomiarowe.
    Każdy efekt stosowany losowo z 50% prawdopodobieństwem.

    Args:
        state: Macierz stanów (steps, 2)
        noise_std: Szum proporcjonalny (3% = 0.03)
        dropout_rate: Odsetek brakujących próbek
        spike_rate: Odsetek skoków/spike'ów
        drift_magnitude: Amplituda dryftu DC
        saturation_clip: Percentyl obcięcia saturacji
    """
    # Szum pomiarowy (zawsze)
    if noise_std > 0:
        state = add_noise(state, noise_std, relative=True)

    # Losowe efekty (każdy z 50% szansą)
    if enable_dropout and np.random.random() < 0.5:
        state = add_dropout(state, dropout_rate)

    if enable_spikes and np.random.random() < 0.3:
        state = add_spikes(state, spike_rate)

    if enable_drift and np.random.random() < 0.4:
        state = add_drift(state, drift_magnitude)

    if enable_saturation and np.random.random() < 0.2:
        clip = np.random.uniform(85, saturation_clip)
        state = add_saturation(state, clip)

    return state


def generate_dataset(
    num_trajectories: int = 100,
    dt: float = 0.01,
    t_max: float = 10.0,
    mass_range: Tuple[float, float] = (1.0, 1.0),
    damping_range: Tuple[float, float] = (0.1, 0.5),
    stiffness_range: Tuple[float, float] = (1.0, 5.0),
    x0_range: Tuple[float, float] = (-2.0, 2.0),
    v0_range: Tuple[float, float] = (-1.0, 1.0),
    noise_std: float = 0.01,
    seed: Optional[int] = None,
    undamped_ratio: float = 0.0
) -> Dict[str, np.ndarray]:
    """
    Generuje zbiór danych zawierający wiele trajektorii oscylatora.

    Każda trajektoria jest generowana z losowymi parametrami fizycznymi
    i warunkami początkowymi z zadanych zakresów.

    Args:
        num_trajectories: Liczba trajektorii do wygenerowania
        dt: Krok czasowy [s]
        t_max: Maksymalny czas symulacji [s]
        mass_range: Zakres masy (min, max) [kg]
        damping_range: Zakres tłumienia (min, max) [Ns/m]
        stiffness_range: Zakres sztywności (min, max) [N/m]
        x0_range: Zakres położenia początkowego (min, max) [m]
        v0_range: Zakres prędkości początkowej (min, max) [m/s]
        noise_std: Odchylenie standardowe szumu pomiarowego
        seed: Ziarno generatora losowego
        undamped_ratio: Proporcja trajektorii bez tłumienia (0.0 - 1.0)

    Returns:
        Słownik zawierający:
            - 'trajectories': Macierz trajektorii (num_traj, num_steps, 2)
            - 'time': Wektor czasu (num_steps,)
            - 'params': Lista parametrów każdej trajektorii
    """
    if seed is not None:
        np.random.seed(seed)

    # Wektor czasu
    t = np.arange(0, t_max, dt)
    num_steps = len(t)

    # Liczba trajektorii bez tłumienia
    num_undamped = int(num_trajectories * undamped_ratio)
    num_damped = num_trajectories - num_undamped

    # Inicjalizacja macierzy trajektorii
    trajectories = np.zeros((num_trajectories, num_steps, 2))
    params_list = []

    # Generowanie trajektorii tłumionych
    for i in range(num_damped):
        mass = np.random.uniform(*mass_range)
        damping = np.random.uniform(*damping_range)
        stiffness = np.random.uniform(*stiffness_range)
        x0 = np.random.uniform(*x0_range)
        v0 = np.random.uniform(*v0_range)

        oscillator = DampedOscillator(
            mass=mass,
            damping=damping,
            stiffness=stiffness
        )

        state = oscillator.generate_state_space(x0, v0, t)

        if noise_std > 0:
            state = add_noise(state, noise_std)

        trajectories[i] = state

        params_list.append({
            'mass': mass,
            'damping': damping,
            'stiffness': stiffness,
            'x0': x0,
            'v0': v0,
            'omega_0': oscillator.params.omega_0,
            'zeta': oscillator.params.zeta,
            'type': 'damped'
        })

    # Generowanie trajektorii bez tłumienia
    for i in range(num_damped, num_trajectories):
        mass = np.random.uniform(*mass_range)
        stiffness = np.random.uniform(*stiffness_range)
        x0 = np.random.uniform(*x0_range)
        v0 = np.random.uniform(*v0_range)

        oscillator = SimpleHarmonicOscillator(
            mass=mass,
            stiffness=stiffness
        )

        state = oscillator.generate_state_space(x0, v0, t)

        if noise_std > 0:
            state = add_noise(state, noise_std)

        trajectories[i] = state

        params_list.append({
            'mass': mass,
            'damping': 0.0,
            'stiffness': stiffness,
            'x0': x0,
            'v0': v0,
            'omega_0': oscillator.params.omega_0,
            'zeta': 0.0,
            'type': 'undamped'
        })

    # Losowe przemieszanie trajektorii
    if undamped_ratio > 0:
        indices = np.random.permutation(num_trajectories)
        trajectories = trajectories[indices]
        params_list = [params_list[i] for i in indices]

    return {
        'trajectories': trajectories,
        'time': t,
        'params': params_list
    }


def generate_dimensionless_dataset(
    num_trajectories: int = 1000,
    dtau: float = 0.1,
    tau_max: float = 50.0,
    zeta_range: Tuple[float, float] = (0.0, 0.5),
    noise_std: float = 0.01,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generuje zbiór danych w parametryzacji bezwymiarowej.

    Każda trajektoria jest generowana z losowym ζ z zadanego zakresu.
    Warunki początkowe są stałe: x(0)=1, dx/dτ(0)=0.
    Dynamika zależy TYLKO od ζ — sieć dostaje wyłącznie [x, dx/dτ].

    Args:
        num_trajectories: Liczba trajektorii do wygenerowania
        dtau: Krok czasowy bezwymiarowy (τ = ω₀·t)
        tau_max: Maksymalny czas bezwymiarowy
        zeta_range: Zakres współczynnika tłumienia (min, max), typowo [0, 0.5]
        noise_std: Odchylenie standardowe szumu pomiarowego
        seed: Ziarno generatora losowego

    Returns:
        Słownik zawierający:
            - 'trajectories': Macierz trajektorii (num_traj, num_steps, 2) - [x, dx/dτ]
            - 'tau': Wektor czasu bezwymiarowego (num_steps,)
            - 'params': Wektor parametrów (num_traj,) - [zeta] (metadane)
    """
    if seed is not None:
        np.random.seed(seed)

    # Wektor czasu bezwymiarowego
    tau = np.arange(0, tau_max, dtau)
    num_steps = len(tau)

    # Inicjalizacja macierzy
    trajectories = np.zeros((num_trajectories, num_steps, 2))
    zeta_array = np.zeros(num_trajectories)

    for i in range(num_trajectories):
        # Losowanie parametru tłumienia
        zeta = np.random.uniform(*zeta_range)

        # Tworzenie oscylatora i generowanie trajektorii
        oscillator = DimensionlessOscillator(zeta=zeta)
        state = oscillator.generate_state_space(tau)

        # Dodanie szumu
        if noise_std > 0:
            state = add_noise(state, noise_std)

        trajectories[i] = state
        zeta_array[i] = zeta

    return {
        'trajectories': trajectories,
        'tau': tau,
        'params': zeta_array,
    }


def quantize_state(state: np.ndarray, num_levels: int) -> np.ndarray:
    """
    Symulacja kwantyzacji cyfrowego czujnika.
    Kwantyzuje dane do zadanej liczby poziomów w zakresie [-max, max].

    Args:
        state: Macierz stanów (steps, 2) — [x, dx/dτ]
        num_levels: Liczba poziomów kwantyzacji (np. 20 = rozdzielczość 0.1 przy amplitudzie 1)

    Returns:
        Macierz stanów po kwantyzacji
    """
    result = state.copy()
    for col in range(state.shape[1]):
        col_max = np.max(np.abs(state[:, col]))
        if col_max > 0:
            step = 2 * col_max / num_levels
            result[:, col] = np.round(state[:, col] / step) * step
    return result


def generate_forced_dimensionless_dataset(
    num_trajectories: int = 960,
    dtau: float = 0.1,
    tau_max: float = 80.0,
    zeta_range: Tuple[float, float] = (0.0, 0.85),
    noise_std: float = 0.03,
    quantize_ratio: float = 0.5,
    onset_ratio: float = 0.3,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generuje zbiór danych oscylatora w parametryzacji bezwymiarowej.

    Zawiera 12 reżimów:
    - 9 wymuszonych: 3 grupy ζ × 3 grupy Ω  (F₀ > 0)
    - 3 swobodne:    3 grupy ζ × Ω=0 (F₀ = 0, drgania gasnące)

    Reżimy wymuszenia (Ω > 0):
        d²x/dτ² + 2ζ(dx/dτ) + x = F₀·cos(Ωτ)
    Reżimy swobodne (Ω = 0):
        d²x/dτ² + 2ζ(dx/dτ) + x = 0

    Args:
        num_trajectories: Liczba trajektorii (zaokrąglana do wielokrotności 12)
        dtau: Krok czasowy bezwymiarowy
        tau_max: Maksymalny czas bezwymiarowy
        zeta_range: Zakres ζ (informacyjny — faktyczne zakresy wynikają z grup)
        noise_std: Odchylenie standardowe szumu pomiarowego
        quantize_ratio: Odsetek trajektorii z symulowaną kwantyzacją cyfrową (0.0–1.0)
        onset_ratio: Odsetek trajektorii z opóźnionym startem wymuszenia (0.0–1.0)
        seed: Ziarno generatora losowego

    Returns:
        Słownik zawierający:
            - 'trajectories': Macierz trajektorii (N, steps, 2) - [x, dx/dτ]
            - 'tau': Wektor czasu bezwymiarowego (steps,)
            - 'params': Macierz parametrów (N, 2) - [zeta, omega] (omega=0 → swobodny)
    """
    if seed is not None:
        np.random.seed(seed)

    # Liczba reżimów: wymuszonych (ζ × Ω) + swobodnych (ζ)
    num_regimes = len(FORCED_ZETA_GROUPS) * len(FORCED_OMEGA_GROUPS) + len(FORCED_ZETA_GROUPS)
    # Zaokrąglenie do wielokrotności num_regimes
    num_trajectories = (num_trajectories // num_regimes) * num_regimes
    per_regime = num_trajectories // num_regimes

    # Wektor czasu bezwymiarowego
    tau = np.arange(0, tau_max, dtau)
    num_steps = len(tau)

    # Inicjalizacja macierzy
    trajectories = np.zeros((num_trajectories, num_steps, 2))
    params_array = np.zeros((num_trajectories, 2))  # [zeta, omega]

    idx = 0
    zeta_keys = sorted(FORCED_ZETA_GROUPS.keys())
    omega_keys = sorted(FORCED_OMEGA_GROUPS.keys())

    # Reżimy wymuszone (ζ × Ω, F₀ > 0)
    for zk in zeta_keys:
        zeta_lo, zeta_hi = FORCED_ZETA_GROUPS[zk]
        for ok in omega_keys:
            omega_lo, omega_hi = FORCED_OMEGA_GROUPS[ok]
            for _ in range(per_regime):
                zeta = np.random.uniform(zeta_lo, zeta_hi)
                omega = np.random.uniform(omega_lo, omega_hi)

                oscillator = ForcedDimensionlessOscillator(
                    zeta=zeta, omega=omega, f0=FORCED_F0
                )

                # Losowo: opóźniony start wymuszenia (okres spoczynku → nagłe wzbudzenie)
                if onset_ratio > 0 and np.random.random() < onset_ratio:
                    tau_onset = np.random.uniform(5.0, tau_max * 0.3)
                    x, v = oscillator.generate_trajectory(tau, tau_onset=tau_onset)
                    state = np.column_stack([x, v])
                else:
                    state = oscillator.generate_state_space(tau)

                # Augmentacja realistycznymi efektami czujnika
                state = apply_sensor_augmentation(state, noise_std=noise_std)

                # Kwantyzacja cyfrowa (losowo)
                if quantize_ratio > 0 and np.random.random() < quantize_ratio:
                    num_levels = np.random.randint(10, 51)
                    state = quantize_state(state, num_levels)

                trajectories[idx] = state
                params_array[idx] = [zeta, omega]
                idx += 1

    # Reżimy swobodne (ζ, F₀ = 0 → drgania gasnące)
    for zk in zeta_keys:
        zeta_lo, zeta_hi = FORCED_ZETA_GROUPS[zk]
        for _ in range(per_regime):
            zeta = np.random.uniform(zeta_lo, zeta_hi)

            oscillator = DimensionlessOscillator(zeta=zeta)
            state = oscillator.generate_state_space(tau)

            # Augmentacja realistycznymi efektami czujnika
            state = apply_sensor_augmentation(state, noise_std=noise_std)

            # Kwantyzacja cyfrowa (losowo)
            if quantize_ratio > 0 and np.random.random() < quantize_ratio:
                num_levels = np.random.randint(10, 51)
                state = quantize_state(state, num_levels)

            trajectories[idx] = state
            params_array[idx] = [zeta, 0.0]
            idx += 1

    # Losowe przemieszanie trajektorii
    perm = np.random.permutation(num_trajectories)
    trajectories = trajectories[perm]
    params_array = params_array[perm]

    return {
        'trajectories': trajectories,
        'tau': tau,
        'params': params_array,
    }


def save_dimensionless_dataset(
    dataset: Dict[str, np.ndarray],
    filepath: str
) -> None:
    """
    Zapisuje zbiór danych bezwymiarowych do pliku .npz.

    Args:
        dataset: Słownik z danymi (z funkcji generate_dimensionless_dataset)
        filepath: Ścieżka do pliku wyjściowego
    """
    np.savez(
        filepath,
        trajectories=dataset['trajectories'],
        tau=dataset['tau'],
        params=dataset['params']
    )


def load_dimensionless_dataset(filepath: str) -> Dict[str, np.ndarray]:
    """
    Wczytuje zbiór danych bezwymiarowych z pliku .npz.

    Args:
        filepath: Ścieżka do pliku wejściowego

    Returns:
        Słownik z danymi
    """
    data = np.load(filepath)
    return {
        'trajectories': data['trajectories'],
        'tau': data['tau'],
        'params': data['params']
    }


def save_dataset(
    dataset: Dict[str, np.ndarray],
    filepath: str
) -> None:
    """
    Zapisuje wygenerowany zbiór danych do pliku .npz.

    Args:
        dataset: Słownik z danymi (z funkcji generate_dataset)
        filepath: Ścieżka do pliku wyjściowego
    """
    # Konwersja listy parametrów na tablicę dla zapisu
    params_array = np.array([
        [p['mass'], p['damping'], p['stiffness'], p['x0'], p['v0']]
        for p in dataset['params']
    ])

    np.savez(
        filepath,
        trajectories=dataset['trajectories'],
        time=dataset['time'],
        params=params_array
    )


def load_dataset(filepath: str) -> Dict[str, np.ndarray]:
    """
    Wczytuje zbiór danych z pliku .npz.

    Args:
        filepath: Ścieżka do pliku wejściowego

    Returns:
        Słownik z danymi
    """
    data = np.load(filepath)
    return {
        'trajectories': data['trajectories'],
        'time': data['time'],
        'params': data['params']
    }


if __name__ == "__main__":
    # Przykładowe użycie - test generatora
    import matplotlib.pyplot as plt

    # Generacja pojedynczej trajektorii
    oscillator = DampedOscillator(mass=1.0, damping=0.3, stiffness=2.0)
    t = np.linspace(0, 10, 1000)
    x, v = oscillator.generate_trajectory(x0=1.0, v0=0.0, t=t)

    print(f"Częstość własna omega_0: {oscillator.params.omega_0:.3f} rad/s")
    print(f"Współczynnik tłumienia zeta: {oscillator.params.zeta:.3f}")

    # Wizualizacja
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Położenie w czasie
    axes[0, 0].plot(t, x, 'b-', linewidth=1)
    axes[0, 0].set_xlabel('Czas [s]')
    axes[0, 0].set_ylabel('Położenie x [m]')
    axes[0, 0].set_title('Położenie vs czas')
    axes[0, 0].grid(True, alpha=0.3)

    # Prędkość w czasie
    axes[0, 1].plot(t, v, 'r-', linewidth=1)
    axes[0, 1].set_xlabel('Czas [s]')
    axes[0, 1].set_ylabel('Prędkość v [m/s]')
    axes[0, 1].set_title('Prędkość vs czas')
    axes[0, 1].grid(True, alpha=0.3)

    # Portret fazowy
    axes[1, 0].plot(x, v, 'g-', linewidth=1)
    axes[1, 0].set_xlabel('Położenie x [m]')
    axes[1, 0].set_ylabel('Prędkość v [m/s]')
    axes[1, 0].set_title('Portret fazowy')
    axes[1, 0].grid(True, alpha=0.3)

    # Generacja datasetu i wizualizacja kilku trajektorii
    dataset = generate_dataset(
        num_trajectories=5,
        dt=0.01,
        t_max=10.0,
        seed=42
    )

    for i in range(5):
        axes[1, 1].plot(
            dataset['time'],
            dataset['trajectories'][i, :, 0],
            alpha=0.7,
            label=f'Trajektoria {i+1}'
        )

    axes[1, 1].set_xlabel('Czas [s]')
    axes[1, 1].set_ylabel('Położenie x [m]')
    axes[1, 1].set_title('Przykładowe trajektorie z datasetu')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/synthetic/oscillator_demo.png', dpi=150)
    plt.show()

    print(f"\nWygenerowano dataset z {dataset['trajectories'].shape[0]} trajektoriami")
    print(f"Kształt danych: {dataset['trajectories'].shape}")
