# Instruction for AI Code Generation – Variant B (Time-Series Prediction)

## Problem Definition
The goal is to design a recurrent neural network (RNN-based) model that predicts **future motion time-series distributions** of a dynamical system based solely on a **past fragment of recorded motion**, without explicit knowledge of physical parameters.

The model should learn temporal dynamics directly from data and provide **probabilistic forecasts** over a future time horizon.

---

## Variant B: Data-Driven Time-Series Prediction

### Input
- A sliding window of past observations:
  - Preferably state-space representation:  
    **[x(t), v(t)]**  
  - Alternatively, raw signal **x(t)** if velocity is unavailable.
- Input window length: `T_in` time steps.
- Data must be uniformly sampled in time.

### Output
- Future trajectory for `T_out` time steps.
- The output should represent a **probability distribution**, not only point estimates.
  - Example outputs:
    - Mean and standard deviation: `(μ(t), σ(t))`
    - Or parameters of a mixture density model (MDN).

---

## Preprocessing Requirements
1. Resample data to a constant time step Δt.
2. Normalize or standardize input features (separately for position and velocity).
3. Apply optional noise filtering (e.g. low-pass or Savitzky–Golay).
4. Construct sliding windows:
   - Input shape: `(batch_size, T_in, features)`
   - Target shape: `(batch_size, T_out, features)`

---

## Model Architecture Recommendations
- Use **LSTM or GRU** (vanilla RNN is not sufficient).
- Prefer **Encoder–Decoder (Seq2Seq)** architecture:
  - Encoder processes the input window.
  - Decoder generates the future trajectory.
- Decoder should support **autoregressive generation**.
- Apply **Layer Normalization** inside recurrent layers if possible.

---

## Long-Horizon Stability Techniques
- Use **multi-step loss** over the entire `T_out` horizon.
- Apply **teacher forcing** with scheduled sampling.
- Use **gradient clipping** (e.g. max norm = 1.0).
- Optimizer: **Adam or AdamW**.

---

## Probabilistic Output & Loss Function
The model should output distribution parameters:
- Gaussian Negative Log-Likelihood (NLL):
  - Output: μ(t), σ(t)
- Optionally: Mixture Density Network (MDN) for multimodal futures.

Loss function must be based on likelihood, not MSE only.

---

## Postprocessing
- Denormalize predictions.
- Compute confidence intervals or uncertainty bands.
- Optionally validate physical plausibility (e.g. damping behavior).

---

## Dimensionless Parameterization Support

The project supports two data modes, detected automatically:

### Dimensional mode (default)
- Time vector key: `time` (units: seconds)
- State vector: `[x(t), v(t)]` with SI units ([m], [m/s])
- Plot labels: `Czas [s]`, `Położenie x [m]`, `Prędkość v [m/s]`
- Predict mode generates both damped and undamped oscillators

### Dimensionless mode (`--dimensionless`)
- Time vector key: `tau` (dimensionless: τ = ω₀·t)
- State vector: `[x(τ), dx/dτ]` (dimensionless)
- Plot labels: `Czas bezwymiarowy τ`, `Położenie x`, `Prędkość dx/dτ`
- Predict mode generates `DimensionlessOscillator` with random ζ from `--zeta-range`
- Fixed initial conditions: x(0) = 1, dx/dτ(0) = 0

### Automatic mode detection
- `visualize_results()` detects mode via `'tau' in dataset` — no flag needed
- All 7 plot functions in `visualization.py` accept optional label parameters with backward-compatible defaults
- `predict()` branches on `args.dimensionless` flag

### Key functions by mode:
| Function | Dimensional | Dimensionless |
|----------|-------------|---------------|
| Data generation | `generate_dataset()` | `generate_dimensionless_dataset()` |
| Data loading | `load_dataset()` | `load_dimensionless_dataset()` |
| Prediction | `run_prediction_for_oscillator()` | `run_prediction_for_dimensionless()` |
| Visualization | Auto-detected labels | Auto-detected labels |
| Model | Same `Seq2SeqModel` | Same `Seq2SeqModel` |

---

## Implementation Constraints
- Language: **Python**
- Framework: **PyTorch**
- Suggested libraries:
  - numpy, scipy
  - torch, pytorch-lightning (optional)
  - matplotlib
  - optuna (optional)

---

## Virtual Environment

⚠️ **IMPORTANT**: Always use the project's virtual environment for running Python code.

```bash
# Activation
source .venv/bin/activate

# Running scripts
python main.py --mode train

# Installing dependencies
pip install -r requirements.txt
```

All Python commands should be executed within the activated virtual environment.

---

## IMPORTANT CODING REQUIREMENT
⚠️ **All code comments, docstrings, and inline explanations MUST be written in Polish**, even though the instruction and code structure are defined in English.

---

## Documentation Language
⚠️ **README.md and all user-facing documentation MUST be written in scientific/academic Polish style**, suitable for an MSc thesis. Avoid colloquial language. Use precise technical terminology.

---

## Expected Output from AI
- Clean, modular PyTorch code.
- Separate modules for:
  - data preprocessing
  - dataset and dataloader
  - model definition
  - training loop
  - evaluation and visualization
- Code should be suitable for inclusion in an engineering or MSc thesis.
- Documentation (README, comments) should use scientific register appropriate for academic work.
