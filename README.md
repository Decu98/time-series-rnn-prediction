# Predykcja Szeregów Czasowych - Model Seq2Seq LSTM

Model rekurencyjnej sieci neuronowej (RNN) do predykcji przyszłych rozkładów ruchu układu dynamicznego na podstawie fragmentu zarejestrowanego ruchu.

## Opis

Projekt implementuje architekturę **Encoder-Decoder (Seq2Seq)** opartą na **LSTM** z probabilistycznym wyjściem Gaussowskim. Model uczy się dynamiki czasowej bezpośrednio z danych i generuje prognozy probabilistyczne (średnia μ i odchylenie standardowe σ) na zadany horyzont czasowy.

### Główne cechy:
- Architektura Seq2Seq z LSTM
- Wyjście probabilistyczne (rozkład Gaussa)
- Teacher forcing ze scheduled sampling
- Gradient clipping dla stabilności
- Integracja z PyTorch Lightning
- **Obsługa dwóch typów oscylatorów:** tłumiony i bez tłumienia
- **Predykcja rekurencyjna** dla długoterminowych prognoz
- Automatyczne testowanie na obu typach oscylatorów

## Wymagania

- Python 3.9+
- PyTorch 2.0+
- PyTorch Lightning 2.0+

## Instalacja

1. **Sklonuj repozytorium:**
```bash
git clone <repository-url>
cd time-series-rnn-prediction
```

2. **Utwórz środowisko wirtualne:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# lub
.venv\Scripts\activate     # Windows
```

3. **Zainstaluj zależności:**
```bash
pip install -r requirements.txt
```

## Struktura projektu

```
├── config/
│   └── config.py              # Konfiguracja hiperparametrów
├── data/
│   └── synthetic/             # Dane syntetyczne
├── src/
│   ├── data_generation/
│   │   └── synthetic.py       # Generator oscylatorów (tłumiony + prosty)
│   ├── preprocessing/
│   │   └── preprocessor.py    # Normalizacja i filtrowanie
│   ├── dataset/
│   │   └── time_series_dataset.py  # Dataset i DataModule
│   ├── models/
│   │   ├── encoder.py         # Encoder LSTM
│   │   ├── decoder.py         # Decoder Gaussowski
│   │   └── seq2seq.py         # Model Seq2Seq (LightningModule)
│   ├── training/
│   │   └── losses.py          # Gaussian NLL Loss
│   └── evaluation/
│       ├── metrics.py         # RMSE, MAE, NLL, CRPS, Coverage
│       └── visualization.py   # Wykresy predykcji
├── outputs/                   # Wyniki treningów
├── main.py                    # Skrypt główny
├── requirements.txt           # Zależności
└── README.md
```

## Użycie

### 1. Generacja danych syntetycznych

Generuje trajektorie oscylatora harmonicznego. Obsługuje dwa typy:
- **Tłumiony** (damped) - oscylacje gasnące
- **Bez tłumienia** (undamped) - oscylacje stałe

```bash
# Tylko trajektorie tłumione (domyślnie)
python main.py --mode generate --num-trajectories 1000 --t-max 20.0 --dt 0.1

# Mieszany dataset (50% tłumione, 50% bez tłumienia)
python main.py --mode generate --num-trajectories 1000 --undamped-ratio 0.5 --t-max 20.0 --dt 0.1
```

**Parametry:**

| Parametr | Domyślnie | Opis |
|----------|-----------|------|
| `--num-trajectories` | 1000 | Liczba trajektorii |
| `--t-max` | 10.0 | Czas symulacji [s] |
| `--dt` | 0.01 | Krok czasowy [s] |
| `--noise-std` | 0.01 | Szum pomiarowy |
| `--undamped-ratio` | 0.0 | Proporcja trajektorii bez tłumienia (0.0-1.0) |

Przy mieszanym datasecie generowane są dwa wykresy przykładowych trajektorii:
- `sample_trajectory_damped.png` - oscylator tłumiony
- `sample_trajectory_undamped.png` - oscylator bez tłumienia

### 2. Trening modelu

```bash
python main.py --mode train
```

**Główne parametry treningu:**

| Parametr | Domyślnie | Opis |
|----------|-----------|------|
| `--max-epochs` | 100 | Maksymalna liczba epok |
| `--batch-size` | 64 | Rozmiar batcha |
| `--learning-rate` | 0.001 | Współczynnik uczenia |
| `--hidden-size` | 64 | Rozmiar warstwy ukrytej LSTM |
| `--num-layers` | 2 | Liczba warstw LSTM |
| `--T-in` | 50 | Długość okna wejściowego |
| `--T-out` | 50 | Horyzont predykcji |
| `--teacher-forcing-ratio` | 0.5 | Początkowy współczynnik TF |
| `--gradient-clip` | 1.0 | Maksymalna norma gradientu |
| `--early-stopping-patience` | 10 | Cierpliwość early stopping |

**Przykład treningu na mieszanym datasecie:**
```bash
python main.py --mode train \
    --max-epochs 100 \
    --T-in 100 \
    --T-out 100 \
    --hidden-size 128 \
    --batch-size 32
```

### 3. Ewaluacja modelu

```bash
python main.py --mode test --checkpoint outputs/run_YYYYMMDD_HHMMSS/checkpoints/best_model*.ckpt
```

Generuje:
- Metryki: RMSE, MAE, NLL, CRPS, Coverage (1σ, 2σ, 3σ)
- Wykresy predykcji z przedziałami ufności
- Portret fazowy
- Ewolucja niepewności

### 4. Predykcja (tryb testowy)

Testuje model na **obu typach oscylatorów** (tłumiony i bez tłumienia) z losowymi parametrami:

```bash
python main.py --mode predict --device cpu \
    --checkpoint outputs/run_*/checkpoints/best.ckpt \
    --T-in 100 --T-out 100 --dt 0.1 --t-max 50
```

**Parametry predykcji:**

| Parametr | Domyślnie | Opis |
|----------|-----------|------|
| `--num-predictions` | 3 | Liczba predykcji na trajektorii |
| `--recursive-steps` | 0 | Liczba kroków rekurencyjnych (0 = wyłączone) |
| `--T-in` | 50 | Długość okna wejściowego |
| `--T-out` | 50 | Horyzont predykcji |
| `--t-max` | 10.0 | Długość trajektorii testowej [s] |
| `--dt` | 0.01 | Krok czasowy [s] |

**Przykład z predykcją rekurencyjną:**
```bash
python main.py --mode predict --device cpu \
    --checkpoint outputs/run_*/checkpoints/best.ckpt \
    --T-in 100 --T-out 100 --dt 0.1 --t-max 60 \
    --num-predictions 3 --recursive-steps 5
```

**Struktura wyników predykcji:**
```
predictions/pred_YYYYMMDD_HHMMSS/
├── parameters.txt
├── damped/                          # Oscylator tłumiony
│   ├── multi_prediction_position.png
│   ├── multi_prediction_velocity.png
│   ├── parameters.txt
│   ├── zooms/                       # Zbliżenia każdej predykcji
│   │   ├── zoom_1_position.png
│   │   ├── zoom_1_velocity.png
│   │   └── ...
│   └── recursive/                   # Predykcja rekurencyjna
│       ├── recursive_position.png
│       └── recursive_velocity.png
└── undamped/                        # Oscylator bez tłumienia
    ├── multi_prediction_position.png
    ├── multi_prediction_velocity.png
    ├── parameters.txt
    ├── zooms/
    └── recursive/
```

## Przykładowy workflow

```bash
# 1. Aktywacja środowiska
source .venv/bin/activate

# 2. Generacja mieszanego datasetu
python main.py --mode generate \
    --num-trajectories 1000 \
    --undamped-ratio 0.5 \
    --t-max 20 --dt 0.1

# 3. Trening modelu
python main.py --mode train \
    --T-in 100 --T-out 100 \
    --max-epochs 100

# 4. Predykcja (testuje oba typy oscylatorów)
python main.py --mode predict --device cpu \
    --checkpoint outputs/run_*/checkpoints/best.ckpt \
    --T-in 100 --T-out 100 --dt 0.1 --t-max 50 \
    --num-predictions 3 --recursive-steps 3
```

## Wyniki

Po treningu wyniki zapisywane są w katalogu `outputs/run_YYYYMMDD_HHMMSS/`:

```
outputs/run_20240131_123456/
├── checkpoints/
│   ├── best_model-epoch=XX-val_loss=X.XXXX.ckpt
│   └── last.ckpt
├── logs/
│   └── version_0/
│       └── metrics.csv
├── plots/
│   ├── prediction_position.png
│   ├── prediction_velocity.png
│   ├── full_trajectory_position.png
│   ├── full_trajectory_velocity.png
│   ├── phase_space.png
│   └── training_curves.png
├── preprocessor_stats.npz
└── metrics.txt
```

## Metryki ewaluacji

| Metryka | Opis |
|---------|------|
| **RMSE** | Root Mean Square Error |
| **MAE** | Mean Absolute Error |
| **NLL** | Negative Log-Likelihood |
| **CRPS** | Continuous Ranked Probability Score |
| **Coverage 1σ** | % próbek w przedziale ±1σ (oczekiwane: 68.27%) |
| **Coverage 2σ** | % próbek w przedziale ±2σ (oczekiwane: 95.45%) |

## Typy oscylatorów

### Oscylator tłumiony (DampedOscillator)
Równanie ruchu: `m·x'' + c·x' + k·x = 0`

- Oscylacje gasnące w czasie
- Portret fazowy: spirala do środka
- Parametry: masa (m), tłumienie (c), sztywność (k)

### Oscylator prosty (SimpleHarmonicOscillator)
Równanie ruchu: `m·x'' + k·x = 0` (c = 0)

- Oscylacje stałe (niegasnące)
- Portret fazowy: zamknięta elipsa
- Parametry: masa (m), sztywność (k)

## Urządzenia obliczeniowe

Program automatycznie wykrywa i wykorzystuje dostępny akcelerator:

| Platforma | System | Akcelerator | Wymagania |
|-----------|--------|-------------|-----------|
| **NVIDIA GPU** | Wszystkie | CUDA | Sterowniki NVIDIA + CUDA Toolkit |
| **AMD GPU** | Linux | ROCm | PyTorch z ROCm |
| **AMD GPU** | Windows | DirectML | torch-directml |
| **Intel GPU** | Windows | DirectML | torch-directml |
| **Apple Silicon** | macOS | MPS | macOS 12.3+ z PyTorch 1.12+ |
| **CPU** | Wszystkie | - | Zawsze dostępne (fallback) |

### Automatyczne wykrywanie (domyślne)
```bash
python main.py --mode train --device auto
```

### Wymuszenie konkretnego urządzenia
```bash
python main.py --mode train --device cpu
python main.py --mode train --device cuda      # NVIDIA / AMD ROCm (Linux)
python main.py --mode train --device mps       # Apple Silicon
python main.py --mode train --device directml  # AMD/Intel na Windows
```

> **Uwaga:** Dla trybu `predict` na Windows z DirectML zalecane jest użycie `--device cpu`
> ze względu na ograniczone wsparcie LSTM w DirectML.

## Rozwiązywanie problemów

### Błąd pamięci GPU
Zmniejsz `batch_size` lub `hidden_size`:
```bash
python main.py --mode train --batch-size 16 --hidden-size 32
```

### Wolny trening
Zwiększ liczbę workerów (jeśli masz wiele rdzeni CPU):
```bash
python main.py --mode train --num-workers 4
```

### Model się nie uczy
- Zmniejsz `learning_rate`
- Zwiększ `teacher_forcing_ratio`
- Sprawdź czy dane są poprawnie znormalizowane

### Błąd DirectML w trybie predict
Użyj CPU dla predykcji:
```bash
python main.py --mode predict --device cpu --checkpoint ...
```

## Licencja

Projekt stworzony na potrzeby pracy dyplomowej.

## Autor

Bartłomiej Dec
