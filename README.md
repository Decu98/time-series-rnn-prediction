# Predykcja szeregow czasowych ukladow dynamicznych z wykorzystaniem modelu Seq2Seq LSTM

Implementacja rekurencyjnej sieci neuronowej typu Encoder-Decoder (Seq2Seq) opartej na architekturze LSTM, przeznaczonej do probabilistycznej predykcji przyszlych trajektorii ukladow dynamicznych na podstawie fragmentu zarejestrowanego ruchu. Model nie wymaga jawnej znajomosci parametrow fizycznych ukladu — dynamike wnioskuje bezposrednio z danych wejsciowych.

## Opis

Projekt realizuje zagadnienie predykcji szeregow czasowych w kontekscie jednowymiarowych ukladow oscylacyjnych. Zaimplementowana architektura Encoder-Decoder z warstwami LSTM generuje prognozy probabilistyczne w postaci parametrow rozkladu Gaussa (wartosc oczekiwana mu(t) oraz odchylenie standardowe sigma(t)) na zadany horyzont czasowy T_out.

### Glowne cechy:
- Architektura Seq2Seq z wielowarstwowym LSTM
- Probabilistyczne wyjscie w postaci parametrow rozkladu normalnego N(mu, sigma^2)
- Funkcja straty oparta na ujemnym logarytmie wiarygodnosci (Gaussian NLL)
- Mechanizm teacher forcing ze scheduled sampling (zanikajacy wykladniczo)
- Obcinanie gradientow (gradient clipping) dla stabilnosci numerycznej
- Pelna integracja z frameworkiem PyTorch Lightning
- **Dwa tryby parametryzacji danych treningowych:**
  - Wymiarowa (m, c, k) — klasyczna parametryzacja fizyczna z jednostkami SI
  - **Bezwymiarowa (zeta)** — uniwersalna parametryzacja eliminujaca redundancje parametrow
- **Obsluga trzech typow oscylatorow:** tlumiony, nietlumiony oraz bezwymiarowy
- **Predykcja rekurencyjna** — iteracyjne generowanie prognoz dlugoterminowych
- **Automatyczna detekcja trybu** — wizualizacja i ewaluacja dostosowuja etykiety osi na podstawie struktury danych (klucz `tau` vs `time`)
- Automatyczne testowanie modelu na roznych konfiguracjach oscylatorow

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

- **Stale warunki poczatkowe:** x(0) = 1, dx/dtau(0) = 0 — eliminacja zmiennosci zwiazanej z amplituda i faza poczatkowa
- **Jednoznaczna zaleznosc:** Przy ustalonych warunkach poczatkowych dynamika ukladu jest calkowicie zdeterminowana przez jeden parametr zeta
- **Wektor stanu wejsciowego:** Siec neuronowa otrzymuje wylacznie [x(tau), dx/dtau] — bez jawnego podawania parametru zeta
- Model wnioskuje wartosc wspolczynnika tlumienia z ksztaltu sekwencji wejsciowej (szybkosc zanikania obwiedni, czestotliwosc oscylacji)

#### Zalety parametryzacji bezwymiarowej w kontekscie uczenia maszynowego

- **Redukcja przestrzeni parametrow:** Zamiast trzech parametrow fizycznych (m, c, k) dynamika opisana jest jednym parametrem bezwymiarowym zeta
- **Eliminacja degeneracji:** Rozne kombinacje (m, c, k) dajace to samo zeta produkuja identyczna dynamike bezwymiarowa — siec nie musi uczyc sie rozrozniac fizycznie rownowaznych ukladow
- **Normalizacja skali:** Stale warunki poczatkowe eliminuja koniecznosc adaptacji do roznych amplitud i faz
- **Uniwersalnosc:** Model wytrenowany na danych bezwymiarowych moze byc stosowany do dowolnego ukladu oscylacyjnego o znanym omega0 (poprzez transformacje tau = omega0*t)

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

Ewaluacja obejmuje:
- Obliczenie metryk na calym zbiorze testowym: RMSE, MAE, NLL, CRPS, Coverage (1sigma, 2sigma, 3sigma)
- Generacja wizualizacji predykcji z przedzialami ufnosci
- Portret fazowy (przestrzen stanow x vs v)
- Wykres ewolucji niepewnosci sigma(t) w horyzoncie predykcji

> **Automatyczna detekcja trybu:** Funkcja wizualizacji automatycznie rozpoznaje tryb bezwymiarowy
> na podstawie obecnosci klucza `tau` w zbiorze danych. Etykiety osi na wykresach
> dostosowywane sa odpowiednio — bez koniecznosci recznej konfiguracji.

### 4. Predykcja na nowych trajektoriach (tryb predict)

Tryb predykcji generuje nowa trajektorie oscylatora (niezalezna od zbioru treningowego), a nastepnie wykonuje serie predykcji modelu w rownomiernie rozmieszczonych punktach trajektorii. Umozliwia to ocene jakosci generalizacji modelu na nieznanych danych.

#### Predykcja w trybie wymiarowym

W trybie wymiarowym program automatycznie testuje model na dwoch konfiguracjach: oscylatorze tlumionym (c > 0) oraz oscylatorze nietlumionym (c = 0), z losowo dobranymi parametrami fizycznymi (m, c, k, x0, v0).

```bash
python main.py --mode predict --device cpu \
    --checkpoint outputs/run_*/checkpoints/best.ckpt \
    --T-in 100 --T-out 100 --dt 0.1 --t-max 50 \
    --num-predictions 3 --recursive-steps 5
```

Wyniki zapisywane sa w podkatalogach `damped/` i `undamped/`.

#### Predykcja w trybie bezwymiarowym

W trybie bezwymiarowym program generuje trajektorie oscylatora bezwymiarowego z losowo dobranym wspolczynnikiem tlumienia zeta z zakresu okreslonego parametrem `--zeta-range`. Warunki poczatkowe sa stale: x(0) = 1, dx/dtau(0) = 0. Wektor czasu bezwymiarowego tau konstruowany jest na podstawie parametrow `--tau-max` i `--dtau`.

```bash
python main.py --mode predict --dimensionless --device cpu \
    --checkpoint outputs/run_*_dimensionless/checkpoints/best*.ckpt \
    --T-in 50 --T-out 30 \
    --tau-max 50.0 --dtau 0.1 \
    --zeta-range 0.0 0.5 \
    --num-predictions 3 --recursive-steps 5
```

Wszystkie generowane wykresy wykorzystuja etykiety bezwymiarowe (os czasu: tau, polozenie: x, predkosc: dx/dtau).

#### Predykcja rekurencyjna

Parametr `--recursive-steps N` (N > 0) uruchamia tryb predykcji rekurencyjnej. Model generuje prognozę na T_out krokow, a nastepnie uzywa wlasnych predykcji jako danych wejsciowych do kolejnego kroku. Proces powtarzany jest N razy, co pozwala na generowanie prognoz na horyzont N * T_out krokow czasowych. Metoda ta umozliwia ocene stabilnosci numerycznej modelu oraz propagacji niepewnosci w dlugoterminowych prognozach.

**Parametry predykcji:**

| Parametr | Domyslnie | Opis |
|----------|-----------|------|
| `--dimensionless` | False | Predykcja z parametryzacja bezwymiarowa |
| `--num-predictions` | 3 | Liczba predykcji w rownomiernie rozmieszczonych punktach trajektorii |
| `--recursive-steps` | 0 | Liczba krokow predykcji rekurencyjnej (0 = wylaczone) |
| `--T-in` | 50 | Dlugosc okna wejsciowego (liczba krokow czasowych) |
| `--T-out` | 50 | Horyzont predykcji (liczba krokow czasowych) |
| `--t-max` | 10.0 | Dlugosc trajektorii testowej [s] (tryb wymiarowy) |
| `--dt` | 0.01 | Krok czasowy [s] (tryb wymiarowy) |
| `--tau-max` | 50.0 | Maksymalny czas bezwymiarowy tau (tryb bezwymiarowy) |
| `--dtau` | 0.1 | Krok czasowy bezwymiarowy (tryb bezwymiarowy) |
| `--zeta-range` | [0.0, 0.5] | Zakres losowania wspolczynnika tlumienia zeta (tryb bezwymiarowy) |

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

# 4. Ewaluacja na zbiorze testowym
python main.py --mode test --dimensionless \
    --data-path data/dimensionless/dataset.npz \
    --checkpoint outputs/run_*_dimensionless/checkpoints/best*.ckpt

# 5. Predykcja na nowej trajektorii (z losowym zeta)
python main.py --mode predict --dimensionless --device cpu \
    --checkpoint outputs/run_*_dimensionless/checkpoints/best*.ckpt \
    --T-in 50 --T-out 30 \
    --tau-max 50.0 --dtau 0.1 \
    --num-predictions 3 --recursive-steps 5
```

## Wyniki

### Struktura katalogu po treningu

Po zakonczeniu treningu wyniki zapisywane sa w katalogu `outputs/run_YYYYMMDD_HHMMSS/` (z sufiksem `_dimensionless` dla trybu bezwymiarowego):

```
outputs/run_20240131_123456_dimensionless/
├── checkpoints/
│   ├── best_model-epoch=XX-val_loss=X.XXXX.ckpt
│   └── last.ckpt
├── logs/
│   └── version_0/
│       └── metrics.csv
├── plots/
│   ├── prediction_position_zoom.png       # Predykcja polozenia z przedzialami ufnosci
│   ├── prediction_velocity_zoom.png       # Predykcja predkosci z przedzialami ufnosci
│   ├── full_trajectory_position.png       # Pelna trajektoria z zaznaczona predykcja
│   ├── full_trajectory_velocity.png       # j.w. dla predkosci
│   ├── prediction_comparison.png          # Porownanie obu cech (x, v)
│   ├── uncertainty_evolution.png          # Ewolucja niepewnosci sigma(t)
│   ├── phase_space.png                    # Portret fazowy (x vs v)
│   └── training_curves.png               # Krzywe uczenia (loss treningowy i walidacyjny)
├── preprocessor_stats.npz                 # Statystyki normalizacji (srednia, odch. std.)
└── metrics.txt                            # Metryki ewaluacji na zbiorze testowym
```

### Struktura katalogu po predykcji (tryb wymiarowy)

```
outputs/predictions/pred_YYYYMMDD_HHMMSS/
├── damped/                                # Oscylator tlumiony
│   ├── multi_prediction_position.png
│   ├── multi_prediction_velocity.png
│   ├── zooms/                             # Powiekszenia poszczegolnych predykcji
│   │   ├── zoom_1_position.png
│   │   └── zoom_1_velocity.png
│   ├── recursive/                         # Predykcja rekurencyjna (jesli --recursive-steps > 0)
│   │   ├── recursive_position.png
│   │   └── recursive_velocity.png
│   └── parameters.txt
├── undamped/                              # Oscylator nietlumiony (analogiczna struktura)
└── parameters.txt                         # Parametry glowne (seed, T_in, T_out, parametry oscylatora)
```

### Struktura katalogu po predykcji (tryb bezwymiarowy)

```
outputs/predictions/pred_YYYYMMDD_HHMMSS/
├── multi_prediction_position.png          # Predykcje polozenia w wielu punktach tau
├── multi_prediction_velocity.png          # Predykcje predkosci bezwymiarowej dx/dtau
├── zooms/                                 # Powiekszenia poszczegolnych predykcji
│   ├── zoom_1_position.png
│   ├── zoom_1_velocity.png
│   └── ...
├── recursive/                             # Predykcja rekurencyjna (jesli --recursive-steps > 0)
│   ├── recursive_position.png
│   └── recursive_velocity.png
└── parameters.txt                         # Parametry (zeta, T_in, T_out, dtau, tau_max)
```

> **Uwaga:** W trybie bezwymiarowym wszystkie wykresy wykorzystuja etykiety bezwymiarowe:
> os czasu — "Czas bezwymiarowy tau", polozenie — "Polozenie x", predkosc — "Predkosc dx/dtau".
> W trybie wymiarowym stosowane sa etykiety z jednostkami SI: "Czas [s]", "Polozenie x [m]", "Predkosc v [m/s]".

## Metryki ewaluacji

Do oceny jakosci predykcji wykorzystywane sa nastepujace metryki, obejmujace zarowno dokladnosc punktowa jak i jakosc estymacji niepewnosci:

| Metryka | Opis | Interpretacja |
|---------|------|---------------|
| **RMSE** | Pierwiastek bledu sredniokwadratowego | Miara dokladnosci punktowej predykcji [jednostki wielkosci fizycznej] |
| **MAE** | Sredni blad bezwzgledny | Bardziej odporna na wartosci odstajace miara bledu punktowego |
| **NLL** | Ujemny logarytm wiarygodnosci (Gaussian) | Miara jakosci estymacji pelnego rozkladu p(y\|x) — im nizsza, tym lepsza kalibracja |
| **CRPS** | Continuous Ranked Probability Score | Metryka probabilistyczna porownujaca predykowany CDF z obserwacja — uogolnienie MAE na rozklady |
| **Coverage 1sigma** | Pokrycie przedzialu ufnosci +-1sigma | Oczekiwana wartosc: 68.27% (1 odchylenie standardowe rozkladu normalnego) |
| **Coverage 2sigma** | Pokrycie przedzialu ufnosci +-2sigma | Oczekiwana wartosc: 95.45% (2 odchylenia standardowe) |

Metryki RMSE i MAE obliczane sa osobno dla kazdej cechy (polozenie x, predkosc v) oraz jako srednia globalna.

## Typy oscylatorow

Projekt implementuje trzy klasy oscylatorow, odpowiadajace roznym postaciom rownania ruchu jednowymiarowego ukladu liniowego.

### Oscylator tlumiony (DampedOscillator)
Rownanie ruchu: `m*x'' + c*x' + k*x = 0`

- Rozwiazanie: oscylacje gasnace eksponencjalnie (dla zeta < 1)
- Portret fazowy: spirala zbiezna do punktu rownowagi (0, 0)
- Parametry fizyczne: masa m [kg], wspolczynnik tlumienia c [Ns/m], sztywnosc k [N/m]
- Warunki poczatkowe: dowolne (x0, v0) — losowane z zadanych zakresow

### Oscylator nietlumiony (SimpleHarmonicOscillator)
Rownanie ruchu: `m*x'' + k*x = 0` (przypadek szczegolny dla c = 0)

- Rozwiazanie: oscylacje harmoniczne o stalej amplitudzie
- Portret fazowy: zamknieta elipsa (ruch okresowy, zachowawczy)
- Parametry fizyczne: masa m [kg], sztywnosc k [N/m]
- Energia calkowita ukladu jest zachowana: E = 0.5*k*x^2 + 0.5*m*v^2 = const

### Oscylator bezwymiarowy (DimensionlessOscillator)
Rownanie ruchu: `d2x/dtau2 + 2*zeta*(dx/dtau) + x = 0`

- Czas bezwymiarowy: tau = omega0*t, gdzie omega0 = sqrt(k/m)
- Stale warunki poczatkowe: x(0) = 1, dx/dtau(0) = 0
- Jedyny parametr sterujacy dynamika: wspolczynnik tlumienia bezwymiarowy zeta ∈ [0, 1)
- Siec neuronowa otrzymuje wylacznie wektor stanu [x, dx/dtau] — bez jawnych parametrow fizycznych
- Umozliwia porownywanie ukladow o roznych parametrach fizycznych, lecz tej samej dynamice bezwymiarowej

## Architektura modelu

### Seq2SeqModel (architektura wspolna dla obu trybow)

Model realizuje architekture Encoder-Decoder (Seq2Seq) z warstwami LSTM:

- **Encoder LSTM:** Przetwarza sekwencje wejsciowa [x(t), v(t)] o dlugosci T_in i koduje ja
  w wektorze stanu ukrytego h ∈ R^(hidden_size). Zastosowano wielowarstwowy LSTM (domyslnie 2 warstwy)
  z regularyzacja dropout miedzy warstwami.
- **Decoder Gaussowski:** Na podstawie stanu ukrytego enkodera generuje autoregresywnie
  sekwencje parametrow rozkladu normalnego (mu_t, sigma_t) dla kazdego kroku t ∈ {1, ..., T_out}.
  Wartosc sigma_t jest wymuszana jako dodatnia poprzez funkcje aktywacji softplus.
- **Wspolna architektura:** Identyczna struktura sieci jest wykorzystywana zarowno dla danych
  wymiarowych [x(t), v(t)] jak i bezwymiarowych [x(tau), dx/dtau]. Roznica polega wylacznie
  na charakterystyce danych treningowych — model sam wnioskuje dynamike z ksztaltu sekwencji wejsciowej.

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
