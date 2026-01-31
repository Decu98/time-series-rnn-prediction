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

## Wymagania

- Python 3.9+
- PyTorch 2.0+
- PyTorch Lightning 2.0+

## Instalacja

1. **Sklonuj repozytorium:**
```bash
git clone <repository-url>
cd Kod
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
│   │   └── synthetic.py       # Generator oscylatora tłumionego
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

Generuje trajektorie oscylatora harmonicznego tłumionego:

```bash
python main.py --mode generate --num-trajectories 1000 --t-max 10.0
```

**Parametry:**
- `--num-trajectories` - liczba trajektorii (domyślnie: 1000)
- `--t-max` - czas symulacji w sekundach (domyślnie: 10.0)
- `--dt` - krok czasowy (domyślnie: 0.01)
- `--noise-std` - szum pomiarowy (domyślnie: 0.01)

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
| `--teacher-forcing-ratio` | 1.0 | Początkowy współczynnik TF |
| `--gradient-clip` | 1.0 | Maksymalna norma gradientu |
| `--early-stopping-patience` | 10 | Cierpliwość early stopping |

**Przykład z niestandardowymi parametrami:**
```bash
python main.py --mode train \
    --max-epochs 200 \
    --hidden-size 128 \
    --num-layers 3 \
    --T-in 100 \
    --T-out 50 \
    --batch-size 32 \
    --learning-rate 0.0005
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

### 4. Predykcja dla nowej trajektorii

```bash
python main.py --mode predict --checkpoint outputs/run_*/checkpoints/best_model*.ckpt
```

## Przykładowy workflow

```bash
# 1. Aktywacja środowiska
source .venv/bin/activate

# 2. Generacja danych (opcjonalne - trening automatycznie generuje dane)
python main.py --mode generate --num-trajectories 1000

# 3. Trening modelu
python main.py --mode train --max-epochs 100

# 4. Ewaluacja na zbiorze testowym
python main.py --mode test --checkpoint outputs/run_*/checkpoints/best_model*.ckpt
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
│   ├── prediction_comparison.png
│   ├── phase_space.png
│   ├── uncertainty_evolution.png
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

## Konfiguracja

Domyślne hiperparametry można modyfikować w `config/config.py`:

```python
@dataclass
class TrainingConfig:
    T_in: int = 50           # Długość okna wejściowego
    T_out: int = 50          # Horyzont predykcji
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 100
    teacher_forcing_ratio: float = 1.0
    teacher_forcing_decay: float = 0.02
    gradient_clip_val: float = 1.0
```

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

### Sprawdzenie dostępnych urządzeń
```python
from config.config import print_device_info
print_device_info()
```

### Instalacja PyTorch dla różnych platform

**NVIDIA (CUDA) - Windows/Linux:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**AMD (ROCm) - Linux:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

**AMD/Intel (DirectML) - Windows:**
```bash
pip install torch torchvision
pip install torch-directml
python main.py --mode train --device directml
```
> **Uwaga:** Program automatycznie używa ręcznej pętli treningowej dla DirectML
> (PyTorch Lightning nie wspiera DirectML). Funkcjonalność jest identyczna.

**Apple Silicon (MPS) - macOS:**
```bash
pip install torch torchvision  # MPS jest domyślnie wspierany
```

**CPU - wszystkie platformy:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

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

## Licencja

Projekt stworzony na potrzeby pracy dyplomowej.

## Autor

Bartłomiej Dec
