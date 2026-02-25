# Predykcja Szeregow Czasowych - Model Seq2Seq LSTM

Model rekurencyjnej sieci neuronowej (RNN) do predykcji przyszlych rozkladow ruchu ukladu dynamicznego na podstawie fragmentu zarejestrowanego ruchu.

## Opis

Projekt implementuje architekture **Encoder-Decoder (Seq2Seq)** oparta na **LSTM** z probabilistycznym wyjsciem Gaussowskim. Model uczy sie dynamiki czasowej bezposrednio z danych i generuje prognozy probabilistyczne (srednia i odchylenie standardowe) na zadany horyzont czasowy.

### Glowne cechy:
- Architektura Seq2Seq z LSTM
- Wyjscie probabilistyczne (rozklad Gaussa)
- Teacher forcing ze scheduled sampling
- Gradient clipping dla stabilnosci
- Integracja z PyTorch Lightning
- **Dwa tryby parametryzacji:**
  - Wymiarowa (m, c, k) - klasyczna parametryzacja fizyczna
  - **Bezwymiarowa (zeta)** - uniwersalna parametryzacja
- **Obsluga dwoch typow oscylatorow:** tlumiony i bez tlumienia
- **Predykcja rekurencyjna** dla dlugoterminowych prognoz
- Automatyczne testowanie na obu typach oscylatorow

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

2. **Utworz srodowisko wirtualne:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# lub
.venv\Scripts\activate     # Windows
```

3. **Zainstaluj zaleznosci:**
```bash
pip install -r requirements.txt
```

## Struktura projektu

```
├── config/
│   └── config.py              # Konfiguracja hiperparametrow
├── data/
│   ├── synthetic/             # Dane syntetyczne (parametryzacja wymiarowa)
│   └── dimensionless/         # Dane syntetyczne (parametryzacja bezwymiarowa)
├── src/
│   ├── data_generation/
│   │   └── synthetic.py       # Generator oscylatorow (tlumiony + prosty + bezwymiarowy)
│   ├── preprocessing/
│   │   └── preprocessor.py    # Normalizacja i filtrowanie
│   ├── dataset/
│   │   └── time_series_dataset.py  # Dataset i DataModule
│   ├── models/
│   │   ├── encoder.py         # Encoder LSTM
│   │   ├── decoder.py         # Decoder Gaussowski
│   │   └── seq2seq.py         # Model Seq2Seq
│   ├── training/
│   │   └── losses.py          # Gaussian NLL Loss
│   └── evaluation/
│       ├── metrics.py         # RMSE, MAE, NLL, CRPS, Coverage
│       └── visualization.py   # Wykresy predykcji
├── outputs/                   # Wyniki treningow
├── main.py                    # Skrypt glowny
├── requirements.txt           # Zaleznosci
└── README.md
```

## Parametryzacja danych

Projekt obsluguje dwa tryby parametryzacji oscylatora:

### 1. Parametryzacja wymiarowa (domyslna)

Klasyczne rownanie ruchu:
```
m*x'' + c*x' + k*x = 0
```

Parametry: masa (m), tlumienie (c), sztywnosc (k), warunki poczatkowe (x0, v0).

**Wady:** Rozne kombinacje parametrow moga dawac podobna dynamike, co utrudnia uczenie.

### 2. Parametryzacja bezwymiarowa (--dimensionless)

Rownanie w czasie bezwymiarowym tau = omega0*t:
```
d2x/dtau2 + 2*zeta*(dx/dtau) + x = 0
```

#### Wyprowadzenie rownania bezwymiarowego

**Krok 1: Rownanie wyjsciowe (wymiarowe)**
```
m*x'' + c*x' + k*x = 0
```

**Krok 2: Dzielimy przez m**
```
x'' + (c/m)*x' + (k/m)*x = 0
```

**Krok 3: Definiujemy parametry bezwymiarowe**
- omega0 = sqrt(k/m) — czestosc wlasna nietlumiona [rad/s]
- zeta = c / (2*sqrt(k*m)) = c / (2*m*omega0) — wspolczynnik tlumienia bezwymiarowy

Stad: c/m = 2*zeta*omega0

**Krok 4: Rownanie przyjmuje postac**
```
x'' + 2*zeta*omega0*x' + omega0^2*x = 0
```

**Krok 5: Wprowadzamy czas bezwymiarowy tau = omega0*t**

Pochodne transformuja sie:
- dx/dt = omega0 * (dx/dtau)
- d2x/dt2 = omega0^2 * (d2x/dtau2)

**Krok 6: Podstawiamy do rownania**
```
omega0^2*(d2x/dtau2) + 2*zeta*omega0*omega0*(dx/dtau) + omega0^2*x = 0
```

**Krok 7: Dzielimy przez omega0^2**
```
d2x/dtau2 + 2*zeta*(dx/dtau) + x = 0
```

#### Rozwiazanie analityczne

Dla zeta < 1 (tlumienie podkrytyczne) rozwiazanie przy warunkach x(0)=1, dx/dtau(0)=0:
```
x(tau) = e^(-zeta*tau) * [cos(omega_d*tau) + (zeta/omega_d)*sin(omega_d*tau)]
```
gdzie omega_d = sqrt(1 - zeta^2) jest znormalizowana czestoscia tlumiona.

#### Kluczowe cechy

- **Stale warunki poczatkowe:** x(0) = 1, dx/dtau(0) = 0
- **Dynamika zalezy TYLKO od zeta** (wspolczynnik tlumienia)
- Siec dostaje wylacznie [x, dx/dtau] — bez dodatkowych parametrow
- Model sam wnioskuje dynamike z sekwencji wejsciowej

#### Zalety parametryzacji bezwymiarowej

- Jeden parametr sterujacy dynamika (zeta)
- Eliminacja zmiennosci zwiazanej z warunkami poczatkowymi
- Siec LSTM uczy sie rozpoznawac tlumienie z ksztaltu trajektorii
- Uniwersalnosc: ta sama dynamika dla roznych ukladow fizycznych o tym samym zeta

## Uzycie

### 1. Generacja danych syntetycznych

#### Dane wymiarowe (klasyczne)

```bash
# Tylko trajektorie tlumione (domyslnie)
python main.py --mode generate --num-trajectories 1000 --t-max 20.0 --dt 0.01

# Mieszany dataset (50% tlumione, 50% bez tlumienia)
python main.py --mode generate --num-trajectories 1000 --undamped-ratio 0.5
```

#### Dane bezwymiarowe (zalecane)

```bash
python main.py --mode generate --dimensionless \
    --num-trajectories 1000 \
    --data-path data/dimensionless/dataset.npz \
    --zeta-range 0.0 0.5 \
    --tau-max 50.0 \
    --dtau 0.1
```

**Parametry generacji danych:**

| Parametr | Domyslnie | Opis |
|----------|-----------|------|
| `--dimensionless` | False | Uzyj parametryzacji bezwymiarowej |
| `--num-trajectories` | 1000 | Liczba trajektorii |
| `--zeta-range` | [0.0, 0.5] | Zakres wspolczynnika tlumienia zeta |
| `--tau-max` | 50.0 | Maksymalny czas bezwymiarowy |
| `--dtau` | 0.1 | Krok czasowy bezwymiarowy |
| `--t-max` | 10.0 | Czas symulacji [s] (tryb wymiarowy) |
| `--dt` | 0.01 | Krok czasowy [s] (tryb wymiarowy) |
| `--noise-std` | 0.01 | Szum pomiarowy |
| `--undamped-ratio` | 0.0 | Proporcja trajektorii bez tlumienia (tryb wymiarowy) |

### 2. Trening modelu

#### Trening z danymi wymiarowymi

```bash
python main.py --mode train \
    --max-epochs 100 \
    --T-in 50 --T-out 50 \
    --hidden-size 64 \
    --batch-size 64
```

#### Trening z danymi bezwymiarowymi (zalecany)

```bash
python main.py --mode train --dimensionless \
    --data-path data/dimensionless/dataset.npz \
    --max-epochs 100 \
    --T-in 50 --T-out 30 \
    --hidden-size 64 \
    --batch-size 64
```

**Glowne parametry treningu:**

| Parametr | Domyslnie | Opis |
|----------|-----------|------|
| `--dimensionless` | False | Trening z parametryzacja bezwymiarowa |
| `--max-epochs` | 100 | Maksymalna liczba epok |
| `--batch-size` | 64 | Rozmiar batcha |
| `--learning-rate` | 0.001 | Wspolczynnik uczenia |
| `--hidden-size` | 64 | Rozmiar warstwy ukrytej LSTM |
| `--num-layers` | 2 | Liczba warstw LSTM |
| `--T-in` | 50 | Dlugosc okna wejsciowego |
| `--T-out` | 50 | Horyzont predykcji |
| `--teacher-forcing-ratio` | 0.5 | Poczatkowy wspolczynnik TF |
| `--teacher-forcing-decay` | 0.05 | Spadek TF na epoke |
| `--gradient-clip` | 1.0 | Maksymalna norma gradientu |
| `--early-stopping-patience` | 10 | Cierpliwosc early stopping |

### 3. Ewaluacja modelu

```bash
# Tryb wymiarowy
python main.py --mode test --checkpoint outputs/run_*/checkpoints/best*.ckpt

# Tryb bezwymiarowy
python main.py --mode test --dimensionless \
    --data-path data/dimensionless/dataset.npz \
    --checkpoint outputs/run_*_dimensionless/checkpoints/best*.ckpt
```

Generuje:
- Metryki: RMSE, MAE, NLL, CRPS, Coverage (1sigma, 2sigma, 3sigma)
- Wykresy predykcji z przedzialami ufnosci
- Portret fazowy
- Ewolucja niepewnosci

### 4. Predykcja (tryb testowy)

Testuje model na nowych trajektoriach:

```bash
# Tryb wymiarowy (testuje oba typy oscylatorow)
python main.py --mode predict --device cpu \
    --checkpoint outputs/run_*/checkpoints/best.ckpt \
    --T-in 100 --T-out 100 --dt 0.1 --t-max 50 \
    --num-predictions 3 --recursive-steps 5
```

**Parametry predykcji:**

| Parametr | Domyslnie | Opis |
|----------|-----------|------|
| `--num-predictions` | 3 | Liczba predykcji na trajektorii |
| `--recursive-steps` | 0 | Liczba krokow rekurencyjnych (0 = wylaczone) |
| `--T-in` | 50 | Dlugosc okna wejsciowego |
| `--T-out` | 50 | Horyzont predykcji |
| `--t-max` | 10.0 | Dlugosc trajektorii testowej [s] |
| `--dt` | 0.01 | Krok czasowy [s] |

## Przykladowe workflow

### Workflow 1: Parametryzacja wymiarowa (klasyczna)

```bash
# 1. Aktywacja srodowiska
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# 2. Generacja mieszanego datasetu
python main.py --mode generate \
    --num-trajectories 1000 \
    --undamped-ratio 0.5 \
    --t-max 20 --dt 0.01

# 3. Trening modelu
python main.py --mode train \
    --T-in 100 --T-out 100 \
    --max-epochs 100

# 4. Predykcja
python main.py --mode predict --device cpu \
    --checkpoint outputs/run_*/checkpoints/best.ckpt \
    --T-in 100 --T-out 100 --dt 0.1 --t-max 50 \
    --num-predictions 3 --recursive-steps 3
```

### Workflow 2: Parametryzacja bezwymiarowa (zalecana)

```bash
# 1. Aktywacja srodowiska
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# 2. Generacja danych bezwymiarowych
python main.py --mode generate --dimensionless \
    --num-trajectories 1000 \
    --data-path data/dimensionless/dataset.npz \
    --zeta-range 0.0 0.5 \
    --tau-max 50.0 --dtau 0.1

# 3. Trening modelu
python main.py --mode train --dimensionless \
    --data-path data/dimensionless/dataset.npz \
    --T-in 50 --T-out 30 \
    --max-epochs 100 \
    --hidden-size 64

# 4. Ewaluacja
python main.py --mode test --dimensionless \
    --data-path data/dimensionless/dataset.npz \
    --checkpoint outputs/run_*_dimensionless/checkpoints/best*.ckpt
```

## Wyniki

Po treningu wyniki zapisywane sa w katalogu `outputs/run_YYYYMMDD_HHMMSS/`:

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
| **Coverage 1sigma** | % probek w przedziale +-1sigma (oczekiwane: 68.27%) |
| **Coverage 2sigma** | % probek w przedziale +-2sigma (oczekiwane: 95.45%) |

## Typy oscylatorow

### Oscylator tlumiony (DampedOscillator)
Rownanie ruchu: `m*x'' + c*x' + k*x = 0`

- Oscylacje gasnace w czasie
- Portret fazowy: spirala do srodka
- Parametry: masa (m), tlumienie (c), sztywnosc (k)

### Oscylator prosty (SimpleHarmonicOscillator)
Rownanie ruchu: `m*x'' + k*x = 0` (c = 0)

- Oscylacje stale (niegasnace)
- Portret fazowy: zamknieta elipsa
- Parametry: masa (m), sztywnosc (k)

### Oscylator bezwymiarowy (DimensionlessOscillator)
Rownanie ruchu: `d2x/dtau2 + 2*zeta*(dx/dtau) + x = 0`

- Czas bezwymiarowy: tau = omega0*t
- Stale warunki poczatkowe: x(0)=1, dx/dtau(0)=0
- Dynamika zalezy tylko od zeta
- Siec dostaje wylacznie [x, dx/dtau] — bez parametrow warunkujacych

## Architektura modelu

### Seq2SeqModel (oba tryby)
- Encoder LSTM: przetwarza sekwencje wejsciowa [x, v]
- Decoder Gaussowski: generuje predykcje probabilistyczne (mu, sigma)
- Ten sam model dla trybu wymiarowego i bezwymiarowego — roznia sie tylko danymi

## Urzadzenia obliczeniowe

Program automatycznie wykrywa i wykorzystuje dostepny akcelerator:

| Platforma | System | Akcelerator | Wymagania |
|-----------|--------|-------------|-----------|
| **NVIDIA GPU** | Wszystkie | CUDA | Sterowniki NVIDIA + CUDA Toolkit |
| **AMD GPU** | Linux | ROCm | PyTorch z ROCm |
| **AMD GPU** | Windows | DirectML | torch-directml |
| **Intel GPU** | Windows | DirectML | torch-directml |
| **Apple Silicon** | macOS | MPS | macOS 12.3+ z PyTorch 1.12+ |
| **CPU** | Wszystkie | - | Zawsze dostepne (fallback) |

### Automatyczne wykrywanie (domyslne)
```bash
python main.py --mode train --device auto
```

### Wymuszenie konkretnego urzadzenia
```bash
python main.py --mode train --device cpu
python main.py --mode train --device cuda      # NVIDIA / AMD ROCm (Linux)
python main.py --mode train --device mps       # Apple Silicon
python main.py --mode train --device directml  # AMD/Intel na Windows
```

> **Uwaga:** Dla trybu `predict` na Windows z DirectML zalecane jest uzycie `--device cpu`
> ze wzgledu na ograniczone wsparcie LSTM w DirectML.

## Rozwiazywanie problemow

### Blad pamieci GPU
Zmniejsz `batch_size` lub `hidden_size`:
```bash
python main.py --mode train --batch-size 16 --hidden-size 32
```

### Wolny trening
Zwieksz liczbe workerow (jesli masz wiele rdzeni CPU):
```bash
python main.py --mode train --num-workers 4
```

### Model sie nie uczy
- Zmniejsz `learning_rate`
- Zwieksz `teacher_forcing_ratio`
- Sprawdz czy dane sa poprawnie znormalizowane

### Blad DirectML w trybie predict
Uzyj CPU dla predykcji:
```bash
python main.py --mode predict --device cpu --checkpoint ...
```

## Licencja

Projekt stworzony na potrzeby pracy dyplomowej.

## Autor

Bartlomiej Dec
