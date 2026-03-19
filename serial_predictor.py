#!/usr/bin/env python3
"""
Program do odczytu danych z portu szeregowego (COM), wizualizacji na bieżąco
oraz predykcji przyszłych wartości za pomocą wytrenowanego modelu Seq2Seq LSTM.

COM wysyła jedną wartość x na linię (położenie w milimetrach).
Program sam mierzy czas, estymuje prędkość, wyznacza ω₀ i x₀ z FFT,
przelicza na postać bezwymiarową i wykonuje predykcję.

Wykresy:
    1. Surowe dane z COM: x(t) [mm]
    2. Surowa prędkość estymowana: v(t) [mm/s]
    3. Dane bezwymiarowe + predykcja modelu (τ, x̃, dx̃/dτ)

Użycie:
    python serial_predictor.py --port COM3 --checkpoint outputs/123x/checkpoints/best.ckpt
    python serial_predictor.py --simulate --checkpoint outputs/123x/checkpoints/best.ckpt
    python serial_predictor.py --simulate --checkpoint outputs/123x/checkpoints/best.ckpt --sim-zeta 0.01 --sim-omega 0.95 --sim-x0 0.005
"""

import argparse
import sys
import time
import threading
import collections
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d

# Dodanie ścieżki projektu do importów
sys.path.insert(0, str(Path(__file__).parent))

from src.models.seq2seq import Seq2SeqModel
from src.preprocessing.preprocessor import DataPreprocessor


def parse_args():
    """Parsowanie argumentów wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description='Odczyt danych z portu szeregowego z predykcją modelu LSTM'
    )

    # Port szeregowy
    parser.add_argument('--port', type=str, default=None,
                        help='Port szeregowy (np. COM3, /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=9600,
                        help='Prędkość transmisji (domyślnie: 9600)')
    parser.add_argument('--separator', type=str, default=',',
                        help='Separator wartości w linii (domyślnie: ",")')
    parser.add_argument('--no-zero', action='store_true',
                        help='Nie wysyłaj "z" (zerowanie) na start')

    # UDP
    parser.add_argument('--udp', type=str, default=None,
                        help='Odczyt przez UDP zamiast COM — podaj ip:port (np. 192.168.1.100:4210)')
    parser.add_argument('--udp-listen-port', type=int, default=4211,
                        help='Port lokalny do nasłuchiwania UDP (domyślnie: 4211)')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Ścieżka do checkpointu modelu (.ckpt)')
    parser.add_argument('--T-in', type=int, default=100,
                        help='Długość okna wejściowego modelu (domyślnie: 100)')
    parser.add_argument('--dtau', type=float, default=0.1,
                        help='Krok bezwymiarowy dτ modelu (domyślnie: 0.1)')

    # Predykcja
    parser.add_argument('--predict-every', type=int, default=50,
                        help='Co ile nowych próbek wykonać predykcję (domyślnie: 50)')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Liczba próbek Monte Carlo (domyślnie: 50)')
    parser.add_argument('--min-amplitude', type=float, default=10.0,
                        help='Minimalna amplituda peak-to-peak [mm] do uruchomienia predykcji (domyślnie: 10.0)')

    # Wizualizacja
    parser.add_argument('--buffer-size', type=int, default=2000,
                        help='Maksymalna liczba próbek w buforze (domyślnie: 2000)')
    parser.add_argument('--update-interval', type=int, default=50,
                        help='Interwał odświeżania wykresu [ms] (domyślnie: 50)')
    parser.add_argument('--display-window', type=float, default=10.0,
                        help='Okno czasowe wykresów 1 i 2 [s] (domyślnie: 10.0)')

    # Tryb symulacji
    parser.add_argument('--simulate', action='store_true',
                        help='Tryb symulacji — generuje dane zamiast czytać z COM')
    parser.add_argument('--sim-zeta', type=float, default=0.05,
                        help='Zeta dla symulacji (domyślnie: 0.05)')
    parser.add_argument('--sim-omega', type=float, default=0.95,
                        help='Omega wymuszenia / omega0 dla symulacji (domyślnie: 0.95)')
    parser.add_argument('--sim-f0', type=float, default=5.0,
                        help='Częstotliwość własna f₀ [Hz] dla symulacji (domyślnie: 5.0)')
    parser.add_argument('--sim-x0', type=float, default=10.0,
                        help='Amplituda początkowa [mm] dla symulacji (domyślnie: 10.0)')

    args = parser.parse_args()
    if not args.simulate and args.udp is None and args.port is None:
        parser.error("--port lub --udp jest wymagany (chyba że użyjesz --simulate)")
    return args


class SerialBuffer:
    """
    Bufor cykliczny na surowe dane z COM: t [s], x [mm].
    Wątek bezpieczny. Przechowuje referencję do portu szeregowego
    dla możliwości wysłania komendy zerowania.
    """

    def __init__(self, max_size: int = 2000):
        self.t_buf = collections.deque(maxlen=max_size)
        self.x_buf = collections.deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.total_received = 0
        self.running = True
        self.serial_port = None   # referencja do portu COM
        self.udp_socket = None    # referencja do socketu UDP
        self.udp_arduino_addr = None  # (ip, port) Arduino

    def add_sample(self, t: float, x: float):
        """Dodaj próbkę (czas, położenie). Odrzuca duplikaty czasu (dt < 1ms)."""
        with self.lock:
            if len(self.t_buf) > 0 and (t - self.t_buf[-1]) < 0.0001:
                return  # duplikat czasu (dt < 0.1ms) — odrzuć
            self.t_buf.append(t)
            self.x_buf.append(x)
            self.total_received += 1

    def get_all(self):
        """Pobierz wszystkie dane jako tablice numpy."""
        with self.lock:
            return np.array(self.t_buf), np.array(self.x_buf)

    def clear(self):
        """Wyczyść bufor (po zerowaniu czujnika)."""
        with self.lock:
            self.t_buf.clear()
            self.x_buf.clear()
            self.total_received = 0

    def send_zero(self):
        """Wyślij komendę 'z' na port COM lub UDP i wyczyść bufor."""
        sent = False
        if self.serial_port is not None:
            try:
                self.serial_port.write(b'z')
                sent = True
            except Exception as e:
                print(f"Błąd wysyłania 'z' (COM): {e}")
        if self.udp_socket is not None and self.udp_arduino_addr is not None:
            try:
                self.udp_socket.sendto(b'z', self.udp_arduino_addr)
                sent = True
            except Exception as e:
                print(f"Błąd wysyłania 'z' (UDP): {e}")
        if sent:
            self.clear()
            print(">>> Wysłano 'z' — zerowanie czujnika, bufor wyczyszczony")
        return sent

    @property
    def count(self):
        with self.lock:
            return len(self.x_buf)


def serial_reader_thread(buffer: SerialBuffer, args):
    """
    Wątek odczytu z COM. Jedna wartość x [mm] na linię.
    Czas mierzony z perf_counter (rozdzielczość <1μs na Windows).
    Na starcie wysyła 'z' (zerowanie czujnika), chyba że --no-zero.
    """
    import serial

    t0 = None

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        buffer.serial_port = ser
        print(f"Otwarty port {args.port} @ {args.baud} baud")
    except Exception as e:
        print(f"BŁĄD: Nie można otworzyć portu {args.port}: {e}")
        buffer.running = False
        return

    # Zerowanie czujnika na starcie
    if not args.no_zero:
        time.sleep(0.5)
        ser.write(b'z')
        print(">>> Wysłano 'z' — zerowanie czujnika")
        time.sleep(0.2)
        ser.reset_input_buffer()

    while buffer.running:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            x = float(line.split(args.separator)[0])
            now = time.perf_counter()
            if t0 is None:
                t0 = now
            buffer.add_sample(now - t0, x)
        except (ValueError, IndexError):
            continue
        except Exception as e:
            print(f"Błąd odczytu: {e}")
            continue

    ser.close()


def udp_reader_thread(buffer: SerialBuffer, args):
    """
    Wątek odczytu z UDP. Arduino wysyła pakiety UDP z wartością x [mm].
    Program nasłuchuje na --udp-listen-port i może wysyłać komendy na ip:port z --udp.
    """
    import socket

    # Parsuj adres Arduino z --udp (ip:port)
    try:
        arduino_host, arduino_port = args.udp.split(':')
        arduino_port = int(arduino_port)
    except ValueError:
        print(f"BŁĄD: Nieprawidłowy format --udp: {args.udp} (oczekiwano ip:port)")
        buffer.running = False
        return

    t0 = None

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', args.udp_listen_port))
        sock.settimeout(1.0)
        buffer.udp_socket = sock
        buffer.udp_arduino_addr = (arduino_host, arduino_port)
        print(f"UDP nasłuchuje na 0.0.0.0:{args.udp_listen_port}")
        print(f"Arduino: {arduino_host}:{arduino_port}")
    except Exception as e:
        print(f"BŁĄD: Nie można otworzyć socketu UDP: {e}")
        buffer.running = False
        return

    # Zerowanie czujnika na starcie
    if not args.no_zero:
        time.sleep(0.3)
        sock.sendto(b'z', (arduino_host, arduino_port))
        print(">>> Wysłano 'z' (UDP) — zerowanie czujnika")
        time.sleep(0.2)

    while buffer.running:
        try:
            data, addr = sock.recvfrom(1024)
            line = data.decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            x = float(line.split(args.separator)[0])
            now = time.perf_counter()
            if t0 is None:
                t0 = now
            buffer.add_sample(now - t0, x)
        except socket.timeout:
            continue
        except (ValueError, IndexError):
            continue
        except Exception as e:
            print(f"Błąd UDP: {e}")
            continue

    sock.close()


def simulation_thread(buffer: SerialBuffer, args):
    """
    Wątek symulacji — generuje dane fizyczne [m, s] oscylatora wymuszonego.
    Równanie: m*x'' + c*x' + k*x = F*cos(ω*t)
    Parametryzacja przez: ω₀ = 2π·f₀, ζ, Ω = ω/ω₀
    """
    from scipy.integrate import solve_ivp

    omega0 = 2 * np.pi * args.sim_f0  # częstotliwość własna [rad/s]
    zeta = args.sim_zeta
    Omega_ratio = args.sim_omega       # Ω = ω/ω₀
    omega_drive = Omega_ratio * omega0
    x0 = args.sim_x0                  # amplituda początkowa [mm]
    f0_force = omega0**2 * x0          # siła wymuszająca skalowana do x₀

    # x'' + 2ζω₀x' + ω₀²x = f₀·cos(ω·t)
    def ode(t, y):
        return [y[1], -2 * zeta * omega0 * y[1] - omega0**2 * y[0] + f0_force * np.cos(omega_drive * t)]

    state = np.array([x0, 0.0])
    t_phys = 0.0
    dt_sim = 0.002  # krok symulacji [s]
    send_every = max(1, int(0.005 / dt_sim))  # ~200 Hz wysyłania
    step = 0

    print(f"Symulacja fizyczna: f₀={args.sim_f0} Hz, ω₀={omega0:.2f} rad/s, "
          f"ζ={zeta}, Ω={Omega_ratio}, x₀={x0} mm")

    while buffer.running:
        sol = solve_ivp(ode, [t_phys, t_phys + dt_sim], state,
                        method='RK45', rtol=1e-10, atol=1e-12)
        state = sol.y[:, -1]
        t_phys += dt_sim
        step += 1

        if step % send_every == 0:
            # Symulacja szumu pomiarowego
            noise = np.random.normal(0, x0 * 0.005)
            buffer.add_sample(t_phys, state[0] + noise)
            time.sleep(0.001)


def estimate_omega0_and_x0(t, x):
    """
    Estymacja ω₀ i x₀ z surowych danych x(t).

    Metoda hybrydowa:
        1. FFT → wstępna estymacja częstotliwości dominującej
        2. scipy.signal.find_peaks → dokładne położenia szczytów
        3. ω₀ z mediany odstępów peak-to-peak (odporniejsze na szum niż FFT)
        4. x₀ z amplitudy pierwszego szczytu (widać wzrost przy dudnieniach)

    Returns:
        omega0: Częstotliwość dominująca [rad/s]
        x0: Amplituda odniesienia [mm]
    """
    from scipy.signal import find_peaks

    n = len(t)
    if n < 10:
        return None, None

    dt_mean = (t[-1] - t[0]) / (n - 1)
    if dt_mean <= 0:
        return None, None

    x_centered = x - np.mean(x)

    # --- Krok 1: FFT → wstępna częstotliwość (do ustalenia min distance) ---
    freqs = np.fft.rfftfreq(n, d=dt_mean)
    fft_mag = np.abs(np.fft.rfft(x_centered))
    fft_mag[0] = 0
    idx_fft_peak = np.argmax(fft_mag)
    f_fft = freqs[idx_fft_peak]

    if f_fft <= 0:
        return None, None

    # --- Krok 2: find_peaks → szczyty i doliny ---
    # Minimalna odległość między szczytami: ~pół okresu z FFT
    min_dist = max(1, int(0.5 / (f_fft * dt_mean)))
    # Minimalna prominencja: 10% rozstępu sygnału (filtracja szumu)
    prominence = (np.max(x_centered) - np.min(x_centered)) * 0.1

    peaks, peak_props = find_peaks(x_centered, distance=min_dist, prominence=prominence)
    valleys, valley_props = find_peaks(-x_centered, distance=min_dist, prominence=prominence)

    # --- Krok 3: ω₀ z odstępów peak-to-peak ---
    if len(peaks) >= 2:
        # Okres z mediany odstępów między szczytami (odporne na outlier)
        peak_periods = np.diff(t[peaks])
        T_period = np.median(peak_periods)
        omega0 = 2 * np.pi / T_period
    else:
        # Fallback na FFT gdy za mało szczytów
        omega0 = 2 * np.pi * f_fft

    # --- Krok 4: x₀ z pierwszego szczytu/doliny ---
    if len(peaks) > 0 and len(valleys) > 0:
        first_peak_val = np.abs(x_centered[peaks[0]])
        first_valley_val = np.abs(x_centered[valleys[0]])
        x0 = max(first_peak_val, first_valley_val)
    elif len(peaks) > 0:
        x0 = np.abs(x_centered[peaks[0]])
    else:
        # Fallback: pierwszy cykl
        period_samples = max(1, int(1.0 / (f_fft * dt_mean)))
        x0 = (np.max(x_centered[:period_samples]) - np.min(x_centered[:period_samples])) / 2.0

    if x0 <= 1e-10:
        x0 = (np.max(x) - np.min(x)) / 2.0
    if x0 <= 1e-10:
        x0 = 1.0

    return omega0, x0


class RealtimePredictor:
    """
    Wizualizacja w czasie rzeczywistym z automatycznym przeliczaniem
    danych fizycznych na bezwymiarowe i predykcją modelu.
    """

    def __init__(self, model, preprocessor, buffer, args):
        self.model = model
        self.preprocessor = preprocessor
        self.buffer = buffer
        self.args = args

        self.T_in = args.T_in
        self.T_out = model.hparams.get('T_out', 50)
        self.dtau = args.dtau
        self.predict_every = args.predict_every

        # Stan predykcji
        self.last_prediction_time = 0.0  # czas ostatniej predykcji (cooldown)
        self.pred_cooldown = 0.0         # minimalny czas do następnej predykcji [s]
        self.est_omega0 = None
        self.est_x0 = None
        # Statystyki dokładności (sumacyjne)
        self.mae_norm = []
        self.mae_mm = []
        self.accuracy_evaluated = set()
        # Dane bezwymiarowe okna wejściowego (ostatnia predykcja)
        self.dim_tau_in = None
        self.dim_x_in = None
        self.dim_v_in = None
        # Ostatnia predykcja bezwymiarowa
        self.dim_tau_pred = None
        self.dim_mu_x = None
        self.dim_sigma_x = None
        self.dim_mu_v = None
        self.dim_sigma_v = None
        # Parametry z momentu ostatniej predykcji
        self.pred_omega0 = None
        self.pred_x0 = None
        self.pred_dc_offset = 0.0
        # Historia predykcji
        self.prediction_history = []
        self.pred_artists_ax1 = []    # linie/fill na wykresie 1 (surowe)
        self.pred_artists_ax2 = []    # linie/fill na wykresie 2 (bezwymiarowe)
        self.pred_colors = plt.cm.rainbow(np.linspace(0, 1, 20))

        # --- Konfiguracja 2 wykresów ---
        self.fig, self.axes = plt.subplots(2, 1, figsize=(14, 8))
        self.fig.suptitle('Predykcja w czasie rzeczywistym — Seq2Seq LSTM   [z = zeruj czujnik]', fontsize=13)

        # Wykres 1: surowe x(t) [mm] — okno 10s + predykcje
        ax1 = self.axes[0]
        self.line_raw_x, = ax1.plot([], [], 'b-', linewidth=0.8, label='x(t) z COM')
        ax1.set_ylabel('Położenie x [mm]')
        ax1.set_xlabel('Czas t [s]')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        self.ax_raw_x = ax1

        # Wykres 2: bezwymiarowy — T_in wejście + predykcje + szara rzeczywistość
        ax2 = self.axes[1]
        self.line_dim_x, = ax2.plot([], [], 'b-', linewidth=1, label='x̃(τ) wejście')
        self.line_dim_actual, = ax2.plot([], [], '-', color='gray', linewidth=1.2,
                                         alpha=0.8, label='x̃(τ) rzeczywistość')
        ax2.set_ylabel('Położenie bezwymiarowe x̃')
        ax2.set_xlabel('Czas bezwymiarowy τ')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        self.ax_dim = ax2

        # Pasek statusu
        self.status_text = self.fig.text(
            0.02, 0.01, '', fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )

    def _compute_velocity(self, t, x):
        """Oblicz prędkość numerycznie z dx/dt (różnice centralne)."""
        v = np.zeros_like(x)
        if len(x) < 3:
            return v
        # Różnice centralne — zabezpieczenie przed dt=0
        dt_fwd = np.diff(t)
        dt_central = t[2:] - t[:-2]
        dt_central[dt_central == 0] = np.nan
        dt_fwd[dt_fwd == 0] = np.nan
        v[1:-1] = (x[2:] - x[:-2]) / dt_central
        v[0] = (x[1] - x[0]) / dt_fwd[0] if not np.isnan(dt_fwd[0]) else 0.0
        v[-1] = (x[-1] - x[-2]) / dt_fwd[-1] if not np.isnan(dt_fwd[-1]) else 0.0
        # Zamień NaN/Inf na 0
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v

    def run_prediction(self, t_raw, x_raw):
        """
        Nowe podejście:
        1. Bierz ostatnie 3s surowych danych
        2. FFT → dominująca częstotliwość (walidacja: jedna dominanta)
        3. ω₀ z find_peaks, x₀ z pierwszego szczytu
        4. τ = ω₀·t → przelicz na bezwymiarowe
        5. Resample do T_in=100 przy dτ=0.1 (τ_span=10)
        6. Predykcja modelem
        """
        from scipy.signal import savgol_filter
        from scipy.interpolate import CubicSpline

        # Bierz ostatnie 3s danych
        analysis_window = 3.0  # [s]
        if len(t_raw) < 20:
            return
        t_now = t_raw[-1]
        mask_3s = t_raw >= (t_now - analysis_window)
        t_window = t_raw[mask_3s].copy()
        x_window_raw = x_raw[mask_3s].copy()

        if len(t_window) < 20:
            return

        # Sprawdzenie minimalnej amplitudy — pomiń szum
        peak_to_peak = np.max(x_window_raw) - np.min(x_window_raw)
        if peak_to_peak < self.args.min_amplitude:
            return

        # Preprocessing: odjęcie DC + Savitzky-Golay
        dc_offset = np.mean(x_window_raw)
        x_window = x_window_raw - dc_offset
        sg_window = min(11, len(x_window) // 2 * 2 - 1)
        if sg_window >= 5:
            x_window = savgol_filter(x_window, sg_window, polyorder=3)

        # 1. Estymacja ω₀ i x₀
        omega0, x0 = estimate_omega0_and_x0(t_window, x_window)
        if omega0 is None or x0 is None:
            return

        # Walidacja: jedna dominanta w FFT
        # Szukamy izolowanych pików (nie sąsiednich binów tego samego piku)
        dt_mean = (t_window[-1] - t_window[0]) / (len(t_window) - 1)
        freqs = np.fft.rfftfreq(len(t_window), d=dt_mean)
        fft_mag = np.abs(np.fft.rfft(x_window - np.mean(x_window)))
        fft_mag[0] = 0
        if len(fft_mag) > 5:
            from scipy.signal import find_peaks as fft_find_peaks
            fft_peaks, _ = fft_find_peaks(fft_mag, distance=3, prominence=np.max(fft_mag) * 0.1)
            if len(fft_peaks) >= 2:
                fft_peak_vals = fft_mag[fft_peaks]
                sorted_peaks = np.sort(fft_peak_vals)[::-1]
                # Odrzuć jeśli drugi izolowany pik > 30% pierwszego
                if sorted_peaks[1] > 0.3 * sorted_peaks[0]:
                    return

        # Walidacja: ω₀ musi być wystarczająco duże by τ_span=10 zmieścić w 3s
        # t_needed = τ_span / ω₀ = (T_in * dτ) / ω₀
        tau_span = self.T_in * self.dtau  # = 10
        t_needed = tau_span / omega0
        if t_needed > analysis_window:
            return  # częstotliwość za niska

        self.est_omega0 = omega0
        self.est_x0 = x0
        self.pred_omega0 = omega0
        self.pred_x0 = x0
        self.pred_dc_offset = dc_offset

        # 2. Przeliczenie na bezwymiarowe
        tau_raw = omega0 * t_window
        x_dim_raw = x_window / x0

        # 3. Resampling do T_in przy dτ — cubic spline
        tau_end = tau_raw[-1]
        tau_start = tau_end - (self.T_in - 1) * self.dtau
        tau_uniform = np.linspace(tau_start, tau_end, self.T_in)

        cs_x = CubicSpline(tau_raw, x_dim_raw, extrapolate=True)
        x_dim_in = cs_x(tau_uniform)
        v_dim_in = cs_x(tau_uniform, 1)  # dx̃/dτ z pochodnej splajnu

        # Zapisz dane bezwymiarowe okna wejściowego (do wykresu)
        self.dim_tau_in = tau_uniform
        self.dim_x_in = x_dim_in
        self.dim_v_in = v_dim_in

        # 4. Normalizacja modelem i predykcja
        state = np.column_stack([x_dim_in, v_dim_in])
        state_norm = self.preprocessor.transform(state)

        input_tensor = torch.tensor(state_norm, dtype=torch.float32)
        with torch.no_grad():
            pred = self.model.predict_trajectory(input_tensor, num_samples=self.args.num_samples)

        mu = pred['mu'].cpu().numpy()
        sigma = pred['sigma'].cpu().numpy()
        mu_denorm, sigma_denorm = self.preprocessor.inverse_transform_gaussian(mu, sigma)

        # Oś czasu predykcji (bezwymiarowa, kontynuacja)
        tau_pred_start = tau_uniform[-1] + self.dtau
        self.dim_tau_pred = np.arange(
            tau_pred_start, tau_pred_start + self.T_out * self.dtau, self.dtau
        )[:self.T_out]
        self.dim_mu_x = mu_denorm[:, 0]
        self.dim_mu_v = mu_denorm[:, 1]
        self.dim_sigma_x = sigma_denorm[:, 0]
        self.dim_sigma_v = sigma_denorm[:, 1]

        # Dorysuj predykcję na 2 wykresach (trwale)
        n = len(self.prediction_history)
        color = self.pred_colors[n % len(self.pred_colors)]

        # Wykres 1: predykcja x [mm] (+ DC offset)
        t_pred_phys = self.dim_tau_pred / omega0
        x_pred_phys = self.dim_mu_x * x0 + self.pred_dc_offset
        sigma_x_phys = self.dim_sigma_x * x0
        line1, = self.ax_raw_x.plot(t_pred_phys, x_pred_phys, '--',
                                     color=color, linewidth=1.2, alpha=0.8)
        fill1 = self.ax_raw_x.fill_between(
            t_pred_phys,
            x_pred_phys - 2 * sigma_x_phys,
            x_pred_phys + 2 * sigma_x_phys,
            alpha=0.1, color=color)
        self.pred_artists_ax1.append((line1, fill1))

        # Wykres 2: predykcja bezwymiarowa τ
        line2, = self.ax_dim.plot(self.dim_tau_pred, self.dim_mu_x, '--',
                                   color=color, linewidth=1.2, alpha=0.8)
        fill2 = self.ax_dim.fill_between(
            self.dim_tau_pred,
            self.dim_mu_x - 2 * self.dim_sigma_x,
            self.dim_mu_x + 2 * self.dim_sigma_x,
            alpha=0.1, color=color)
        self.pred_artists_ax2.append((line2, fill2))

        # Zapisz do historii
        self.prediction_history.append({
            'tau_pred': self.dim_tau_pred.copy(),
            't_pred_phys': t_pred_phys.copy(),
            'mu_x': self.dim_mu_x.copy(),
            'x_pred_mm': x_pred_phys.copy(),
            'sigma_x': self.dim_sigma_x.copy(),
            'omega0': omega0,
            'x0': x0,
            'dc_offset': self.pred_dc_offset,
            't_input_end': t_window[-1],
        })

        # Cooldown: nie predykuj dopóki nie minie połowa horyzontu fizycznego
        pred_horizon_phys = (self.T_out * self.dtau) / omega0
        self.pred_cooldown = pred_horizon_phys / 2.0
        self.last_prediction_time = time.perf_counter()

        # Ogranicz do 10 ostatnich predykcji — usuń najstarszą
        max_history = 10
        if len(self.prediction_history) > max_history:
            self.prediction_history.pop(0)
            for artists in [self.pred_artists_ax1, self.pred_artists_ax2]:
                old_line, old_fill = artists.pop(0)
                old_line.remove()
                old_fill.remove()

    def update(self, frame):
        """Callback animacji — 2 wykresy: surowe x(t) + bezwymiarowe z predykcją."""
        if self.buffer.count == 0:
            return []

        t_raw, x_raw = self.buffer.get_all()

        # Wykres 1: surowe x(t) — okno display_window sekund
        win = self.args.display_window
        t_now = t_raw[-1] if len(t_raw) > 0 else 0
        t_win_start = t_now - win
        mask_win = t_raw >= t_win_start
        self.line_raw_x.set_data(t_raw[mask_win], x_raw[mask_win])
        self._autoscale_window(self.ax_raw_x, t_win_start, t_now, x_raw[mask_win])

        # Czy czas na predykcję? (3s danych + cooldown)
        has_enough_data = len(t_raw) > 1 and (t_raw[-1] - t_raw[0]) >= 3.0
        cooldown_ok = (time.perf_counter() - self.last_prediction_time) >= self.pred_cooldown
        if has_enough_data and cooldown_ok:
            self.run_prediction(t_raw, x_raw)

        # Wykres 2: bezwymiarowy — T_in wejście + szara rzeczywistość
        if self.dim_tau_in is not None:
            self.line_dim_x.set_data(self.dim_tau_in, self.dim_x_in)

        if self.dim_tau_in is not None and self.pred_omega0 is not None and len(t_raw) > 0:
            tau_all = self.pred_omega0 * t_raw
            x_dim_all = (x_raw - self.pred_dc_offset) / self.pred_x0
            tau_input_end = self.dim_tau_in[-1]
            mask = tau_all > tau_input_end
            if np.any(mask):
                self.line_dim_actual.set_data(tau_all[mask], x_dim_all[mask])
            else:
                self.line_dim_actual.set_data([], [])

            # Autoscale wykresu 2
            tau_current = self.pred_omega0 * t_raw[-1]
            tau_left = self.dim_tau_in[0]
            tau_right = tau_current
            for ph in self.prediction_history:
                tau_right = max(tau_right, ph['tau_pred'][-1])
            margin_t = (tau_right - tau_left) * 0.03 + 0.5
            self.ax_dim.set_xlim(tau_left - margin_t, tau_right + margin_t)
            self.ax_dim.relim()
            self.ax_dim.autoscale_view(scalex=False, scaley=True)

        # Ewaluacja dokładności — tylko najnowsza nieoceniona predykcja
        if len(t_raw) > 0 and len(self.prediction_history) > 0:
            ph = self.prediction_history[-1]
            pred_id = id(ph)
            if pred_id not in self.accuracy_evaluated:
                t_pred_end = ph['t_pred_phys'][-1]
                if t_raw[-1] >= t_pred_end:
                    t_pred = ph['t_pred_phys']
                    x0 = ph['x0']
                    mask_range = (t_raw >= t_pred[0]) & (t_raw <= t_pred[-1])
                    if np.sum(mask_range) >= 2:
                        interp_actual = interp1d(t_raw[mask_range], x_raw[mask_range],
                                                 kind='linear', fill_value='extrapolate')
                        x_actual_mm = interp_actual(t_pred)
                        self.mae_mm.append(np.mean(np.abs(ph['x_pred_mm'] - x_actual_mm)))
                        self.mae_norm.append(np.mean(np.abs(ph['mu_x'] - x_actual_mm / x0)))
                    self.accuracy_evaluated.add(pred_id)

        # Status
        n_evaluated = len(self.mae_norm)
        if len(t_raw) > 1 and (t_raw[-1] - t_raw[0]) >= 3.0:
            mask_3s = t_raw >= (t_raw[-1] - 3.0)
            p2p = np.max(x_raw[mask_3s]) - np.min(x_raw[mask_3s])
            vibr_status = f"p2p={p2p:.1f}mm" if p2p >= self.args.min_amplitude else f"p2p={p2p:.1f}mm (szum)"
        else:
            t_elapsed = t_raw[-1] - t_raw[0] if len(t_raw) > 1 else 0
            vibr_status = f"czekam... ({t_elapsed:.1f}/3.0s)"
        status = f"{vibr_status}  |  Predykcji: {len(self.prediction_history)}  Ocenionych: {n_evaluated}"
        if self.est_omega0 is not None:
            f0_est = self.est_omega0 / (2 * np.pi)
            status += f"  |  f₀≈{f0_est:.2f}Hz  x₀≈{self.est_x0:.1f}mm  cooldown≈{self.pred_cooldown:.2f}s"
        if n_evaluated > 0:
            status += f"  |  MAE: {np.mean(self.mae_mm):.1f}mm ({np.mean(self.mae_norm):.3f} x/x₀)"
        self.status_text.set_text(status)

        return []

    def _autoscale_window(self, ax, t_start, t_end, y_data):
        """Skalowanie osi z ustalonym oknem czasowym."""
        margin = (t_end - t_start) * 0.02
        ax.set_xlim(t_start - margin, t_end + margin)
        if len(y_data) > 0:
            y_clean = y_data[np.isfinite(y_data)]
            if len(y_clean) == 0:
                return
            y_min, y_max = np.min(y_clean), np.max(y_clean)
            y_margin = (y_max - y_min) * 0.1 + abs(y_max) * 0.01 + 1e-6
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

    def _autoscale_range(self, ax, t_data, y_min_data, y_max_data):
        """Automatyczne skalowanie z osobnymi min/max dla y."""
        if len(t_data) == 0:
            return
        t_margin = (t_data[-1] - t_data[0]) * 0.03 + 0.1
        ax.set_xlim(t_data[0] - t_margin, t_data[-1] + t_margin)
        y_min, y_max = np.min(y_min_data), np.max(y_max_data)
        y_margin = (y_max - y_min) * 0.1 + abs(y_max) * 0.01 + 1e-6
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    def _autoscale(self, ax, t_data, y_data):
        """Automatyczne skalowanie osi."""
        if len(t_data) == 0:
            return
        t_margin = (t_data[-1] - t_data[0]) * 0.03 + 0.1
        ax.set_xlim(t_data[0] - t_margin, t_data[-1] + t_margin)
        y_min, y_max = np.min(y_data), np.max(y_data)
        y_margin = (y_max - y_min) * 0.1 + abs(y_max) * 0.01 + 1e-6
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    def _on_key(self, event):
        """Obsługa klawiszy — 'z' zeruje czujnik i czyści bufor."""
        if event.key == 'z':
            if self.buffer.send_zero():
                # Reset stanu predykcji
                self.last_prediction_time = 0.0
                self.pred_cooldown = 0.0
                self.est_omega0 = None
                self.est_x0 = None
                self.dim_tau_in = None
                self.dim_tau_pred = None
                self.pred_omega0 = None
                self.pred_x0 = None
                # Wyczyść wykresy
                self.line_raw_x.set_data([], [])
                self.line_dim_x.set_data([], [])
                self.line_dim_actual.set_data([], [])
                # Usuń trwałe linie predykcji
                for artists in [self.pred_artists_ax1, self.pred_artists_ax2]:
                    for line, fill in artists:
                        line.remove()
                        fill.remove()
                    artists.clear()
                self.prediction_history.clear()
                self.mae_norm.clear()
                self.mae_mm.clear()
                self.accuracy_evaluated.clear()

    def start(self):
        """Uruchomienie animacji. Klawisz 'z' zeruje czujnik."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.ani = animation.FuncAnimation(
            self.fig, self.update,
            interval=self.args.update_interval,
            blit=False, cache_frame_data=False
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()


def main():
    args = parse_args()

    # Ładowanie modelu
    print(f"Ładowanie modelu z: {args.checkpoint}")
    model = Seq2SeqModel.load_from_checkpoint(args.checkpoint, map_location='cpu')
    model.eval()

    T_out = model.hparams.get('T_out', 50)
    print(f"Parametry modelu: T_in={args.T_in}, T_out={T_out}, dτ={args.dtau}")

    # Ładowanie preprocessora
    preprocessor = DataPreprocessor()
    checkpoint_dir = Path(args.checkpoint).parent.parent
    preprocessor_path = checkpoint_dir / 'preprocessor_stats.npz'

    if preprocessor_path.exists():
        preprocessor.load_stats(str(preprocessor_path))
        print(f"Załadowano preprocessor z: {preprocessor_path}")
    else:
        print(f"UWAGA: Brak pliku preprocessora w {preprocessor_path}")
        sys.exit(1)

    # Bufor danych
    buffer = SerialBuffer(max_size=args.buffer_size)

    # Uruchomienie wątku odczytu
    if args.simulate:
        print(f"\n--- TRYB SYMULACJI ---")
        print(f"  f₀={args.sim_f0} Hz, ζ={args.sim_zeta}, "
              f"Ω={args.sim_omega}, x₀={args.sim_x0} mm")
        reader = threading.Thread(
            target=simulation_thread, args=(buffer, args), daemon=True
        )
    elif args.udp:
        print(f"\n--- TRYB UDP ---")
        print(f"  Arduino: {args.udp}, nasłuch: 0.0.0.0:{args.udp_listen_port}")
        reader = threading.Thread(
            target=udp_reader_thread, args=(buffer, args), daemon=True
        )
    else:
        try:
            import serial  # noqa: F401
        except ImportError:
            print("BŁĄD: Brak biblioteki pyserial. Zainstaluj: pip install pyserial")
            sys.exit(1)
        print(f"\nOdczyt z: {args.port} @ {args.baud} baud")
        reader = threading.Thread(
            target=serial_reader_thread, args=(buffer, args), daemon=True
        )

    reader.start()

    predictor = RealtimePredictor(model, preprocessor, buffer, args)

    try:
        predictor.start()
    except KeyboardInterrupt:
        pass
    finally:
        buffer.running = False
        print("\nZakończono.")


if __name__ == '__main__':
    main()
