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


def add_noise(
    data: np.ndarray,
    noise_std: float = 0.01,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Dodaje szum gaussowski do danych.

    Args:
        data: Dane wejściowe (dowolny kształt)
        noise_std: Odchylenie standardowe szumu
        seed: Ziarno generatora losowego (opcjonalne)

    Returns:
        Dane z dodanym szumem
    """
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.normal(0, noise_std, data.shape)
    return data + noise


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
