#!/usr/bin/env python3
"""
Główny skrypt projektu predykcji szeregów czasowych.

Realizuje pełny pipeline:
1. Generacja/ładowanie danych syntetycznych
2. Preprocessing i normalizacja
3. Tworzenie DataModule
4. Trening modelu Seq2Seq (LSTM)
5. Ewaluacja i wizualizacja wyników

Użycie:
    python main.py --mode train
    python main.py --mode test --checkpoint checkpoints/best_model.ckpt
    python main.py --mode predict --checkpoint checkpoints/best_model.ckpt
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

# Import modułów projektu
from config.config import (
    Config,
    get_default_config,
    get_device,
    get_accelerator_config,
    print_device_info,
    is_directml_available,
)
from src.data_generation.synthetic import generate_dataset, save_dataset, load_dataset
from src.preprocessing.preprocessor import DataPreprocessor
from src.dataset.time_series_dataset import TimeSeriesDataModule
from src.models.seq2seq import Seq2SeqModel
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.visualization import (
    plot_prediction_with_uncertainty,
    plot_training_curves,
    plot_phase_space,
    plot_prediction_comparison,
    plot_uncertainty_evolution,
    plot_full_trajectory_with_prediction,
    plot_multi_prediction_trajectory,
    plot_recursive_prediction,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parsuje argumenty wiersza poleceń.

    Returns:
        Namespace z argumentami
    """
    parser = argparse.ArgumentParser(
        description='Predykcja szeregów czasowych z użyciem modelu Seq2Seq LSTM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Tryb działania
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'predict', 'generate'],
        default='train',
        help='Tryb działania programu'
    )

    # Ścieżki
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/synthetic/dataset.npz',
        help='Ścieżka do pliku z danymi'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Ścieżka do checkpointu modelu (do wznowienia treningu lub ewaluacji)'
    )
    parser.add_argument(
        '--num-predictions',
        type=int,
        default=3,
        help='Liczba predykcji na trajektorii (tryb predict)'
    )
    parser.add_argument(
        '--recursive-steps',
        type=int,
        default=0,
        help='Liczba kroków rekurencyjnych (0=wyłączony, >0=predykcja rekurencyjna)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Katalog na wyniki i wykresy'
    )

    # Parametry danych
    parser.add_argument(
        '--num-trajectories',
        type=int,
        default=1000,
        help='Liczba trajektorii do wygenerowania'
    )
    parser.add_argument(
        '--t-max',
        type=float,
        default=10.0,
        help='Maksymalny czas symulacji [s]'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.01,
        help='Krok czasowy [s]'
    )
    parser.add_argument(
        '--noise-std',
        type=float,
        default=0.01,
        help='Odchylenie standardowe szumu pomiarowego'
    )

    # Parametry okien czasowych
    parser.add_argument(
        '--T-in',
        type=int,
        default=50,
        help='Długość okna wejściowego (historia)'
    )
    parser.add_argument(
        '--T-out',
        type=int,
        default=50,
        help='Długość horyzontu predykcji'
    )

    # Parametry modelu
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=64,
        help='Rozmiar warstwy ukrytej LSTM'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Liczba warstw LSTM'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Współczynnik dropout'
    )

    # Parametry treningu
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Rozmiar batcha'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Współczynnik uczenia'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=100,
        help='Maksymalna liczba epok'
    )
    parser.add_argument(
        '--teacher-forcing-ratio',
        type=float,
        default=0.5,
        help='Początkowy współczynnik teacher forcing (0.5 = 50%% szans na użycie prawdziwej wartości)'
    )
    parser.add_argument(
        '--teacher-forcing-decay',
        type=float,
        default=0.05,
        help='Spadek teacher forcing na epokę (szybszy decay = lepsza generalizacja)'
    )
    parser.add_argument(
        '--gradient-clip',
        type=float,
        default=1.0,
        help='Maksymalna norma gradientu'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=10,
        help='Cierpliwość dla early stopping'
    )

    # Inne
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Ziarno generatora losowego'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps', 'directml'],
        help='Urządzenie obliczeniowe (directml dla AMD na Windows)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Liczba procesów do ładowania danych (0 = główny wątek, zalecane 4-8 dla GPU)'
    )

    return parser.parse_args()


def setup_seed(seed: int) -> None:
    """
    Ustawia ziarno generatora losowego dla reprodukowalności.

    Args:
        seed: Wartość ziarna
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def get_accelerator(device: str) -> str:
    """
    Określa akcelerator dla PyTorch Lightning.

    Obsługiwane platformy:
    - NVIDIA GPU (CUDA)
    - AMD GPU (ROCm - używa CUDA API)
    - Apple Silicon (MPS)
    - CPU

    Args:
        device: Wybrane urządzenie ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        Nazwa akceleratora dla pl.Trainer
    """
    config = get_accelerator_config(device)
    return config["accelerator"]


def generate_data(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    """
    Generuje dane syntetyczne i zapisuje do pliku.

    Args:
        args: Argumenty wiersza poleceń

    Returns:
        Słownik z wygenerowanymi danymi
    """
    print("\n" + "=" * 60)
    print("GENERACJA DANYCH SYNTETYCZNYCH")
    print("=" * 60)

    # Tworzenie katalogu jeśli nie istnieje
    data_dir = Path(args.data_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generowanie danych
    print(f"\nGenerowanie {args.num_trajectories} trajektorii...")
    print(f"  - Czas symulacji: {args.t_max} s")
    print(f"  - Krok czasowy: {args.dt} s")
    print(f"  - Szum pomiarowy: σ = {args.noise_std}")

    dataset = generate_dataset(
        num_trajectories=args.num_trajectories,
        dt=args.dt,
        t_max=args.t_max,
        noise_std=args.noise_std,
        seed=args.seed
    )

    # Zapisanie do pliku
    save_dataset(dataset, args.data_path)
    print(f"\nDane zapisane do: {args.data_path}")
    print(f"Kształt danych: {dataset['trajectories'].shape}")

    # Wizualizacja losowej trajektorii
    plot_sample_trajectory(dataset, data_dir)

    return dataset


def plot_sample_trajectory(dataset: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Generuje wykres losowej trajektorii z datasetu.

    Args:
        dataset: Słownik z danymi (trajectories, time, params)
        output_dir: Katalog na zapis wykresu
    """
    import matplotlib.pyplot as plt

    # Losowy indeks trajektorii
    idx = np.random.randint(0, dataset['trajectories'].shape[0])
    trajectory = dataset['trajectories'][idx]
    time = dataset['time']
    params = dataset['params'][idx] if isinstance(dataset['params'], list) else dataset['params'][idx]

    x = trajectory[:, 0]  # położenie
    v = trajectory[:, 1]  # prędkość

    # Tworzenie wykresu 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Przykładowa trajektoria #{idx} z wygenerowanego datasetu', fontsize=14, fontweight='bold')

    # 1. Położenie w czasie
    axes[0, 0].plot(time, x, 'b-', linewidth=1.2)
    axes[0, 0].set_xlabel('Czas [s]')
    axes[0, 0].set_ylabel('Położenie x [m]')
    axes[0, 0].set_title('Położenie vs czas')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # 2. Prędkość w czasie
    axes[0, 1].plot(time, v, 'r-', linewidth=1.2)
    axes[0, 1].set_xlabel('Czas [s]')
    axes[0, 1].set_ylabel('Prędkość v [m/s]')
    axes[0, 1].set_title('Prędkość vs czas')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # 3. Portret fazowy
    axes[1, 0].plot(x, v, 'g-', linewidth=1.2)
    axes[1, 0].plot(x[0], v[0], 'go', markersize=10, label='Start')
    axes[1, 0].plot(x[-1], v[-1], 'rs', markersize=10, label='Koniec')
    axes[1, 0].set_xlabel('Położenie x [m]')
    axes[1, 0].set_ylabel('Prędkość v [m/s]')
    axes[1, 0].set_title('Portret fazowy (przestrzeń stanów)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    # 4. Informacje o parametrach
    axes[1, 1].axis('off')

    # Wyciągnięcie parametrów
    if isinstance(params, np.ndarray):
        mass, damping, stiffness, x0, v0 = params[:5]
        omega_0 = np.sqrt(stiffness / mass)
        zeta = damping / (2 * np.sqrt(stiffness * mass))
    else:
        mass = params['mass']
        damping = params['damping']
        stiffness = params['stiffness']
        x0 = params['x0']
        v0 = params['v0']
        omega_0 = params.get('omega_0', np.sqrt(stiffness / mass))
        zeta = params.get('zeta', damping / (2 * np.sqrt(stiffness * mass)))

    info_text = f"""
    PARAMETRY OSCYLATORA TŁUMIONEGO
    ════════════════════════════════════

    Równanie ruchu:
    m·x'' + c·x' + k·x = 0

    Parametry fizyczne:
    ─────────────────────────────────────
    Masa (m):              {mass:.4f} kg
    Tłumienie (c):         {damping:.4f} Ns/m
    Sztywność (k):         {stiffness:.4f} N/m

    Warunki początkowe:
    ─────────────────────────────────────
    Położenie (x₀):        {x0:.4f} m
    Prędkość (v₀):         {v0:.4f} m/s

    Parametry charakterystyczne:
    ─────────────────────────────────────
    Częstość własna (ω₀): {omega_0:.4f} rad/s
    Wsp. tłumienia (ζ):    {zeta:.4f}

    Typ ruchu: {'Podkrytyczny (oscylacje)' if zeta < 1 else 'Nadkrytyczny (brak oscylacji)' if zeta > 1 else 'Krytyczny'}
    """

    axes[1, 1].text(0.1, 0.95, info_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Zapis wykresu
    plot_path = output_dir / 'sample_trajectory.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nWykres przykładowej trajektorii zapisany do: {plot_path}")


def load_or_generate_data(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    """
    Ładuje dane z pliku lub generuje nowe jeśli plik nie istnieje.

    Args:
        args: Argumenty wiersza poleceń

    Returns:
        Słownik z danymi
    """
    if Path(args.data_path).exists():
        print(f"\nŁadowanie danych z: {args.data_path}")
        dataset = load_dataset(args.data_path)
        print(f"Załadowano dane o kształcie: {dataset['trajectories'].shape}")
        # Wizualizacja przykładowej trajektorii
        data_dir = Path(args.data_path).parent
        plot_sample_trajectory(dataset, data_dir)
    else:
        print(f"\nPlik {args.data_path} nie istnieje - generowanie nowych danych...")
        dataset = generate_data(args)

    return dataset


def create_datamodule(
    trajectories: np.ndarray,
    args: argparse.Namespace,
    preprocessor: DataPreprocessor
) -> TimeSeriesDataModule:
    """
    Tworzy DataModule z przetworzonymi danymi.

    Args:
        trajectories: Surowe trajektorie
        args: Argumenty wiersza poleceń
        preprocessor: Dopasowany preprocessor

    Returns:
        TimeSeriesDataModule
    """
    # Normalizacja danych
    normalized = preprocessor.transform(trajectories)

    # Tworzenie DataModule
    datamodule = TimeSeriesDataModule(
        data=normalized,
        T_in=args.T_in,
        T_out=args.T_out,
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        stride=1,
        num_workers=args.num_workers,
        seed=args.seed
    )

    return datamodule


def create_model(args: argparse.Namespace) -> Seq2SeqModel:
    """
    Tworzy model Seq2Seq.

    Args:
        args: Argumenty wiersza poleceń

    Returns:
        Model Seq2Seq
    """
    model = Seq2SeqModel(
        input_size=2,  # [x, v]
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        T_out=args.T_out,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        teacher_forcing_decay=args.teacher_forcing_decay,
        gradient_clip_val=args.gradient_clip
    )

    return model


def create_trainer(args: argparse.Namespace, output_dir: Path) -> pl.Trainer:
    """
    Tworzy PyTorch Lightning Trainer z callbackami.

    Args:
        args: Argumenty wiersza poleceń
        output_dir: Katalog na wyniki

    Returns:
        Trainer
    """
    # Callbacks
    callbacks = [
        # Zapis najlepszego modelu
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='best_model-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True,
            verbose=True
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            mode='min',
            verbose=True
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval='epoch')
    ]

    # Logger
    logger = CSVLogger(
        save_dir=output_dir,
        name='logs'
    )

    # Akcelerator
    accelerator = get_accelerator(args.device)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip,
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=True
    )

    return trainer


def train(args: argparse.Namespace) -> None:
    """
    Przeprowadza trening modelu z użyciem PyTorch Lightning.

    Args:
        args: Argumenty wiersza poleceń
    """
    print("\n" + "=" * 60)
    print("TRENING MODELU")
    print("=" * 60)

    # Ustawienie ziarna
    setup_seed(args.seed)

    # Tworzenie katalogu wyjściowego
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nKatalog wyjściowy: {output_dir}")

    # Ładowanie/generowanie danych
    dataset = load_or_generate_data(args)
    trajectories = dataset['trajectories']

    # Preprocessing
    print("\nPreprocessing danych...")
    preprocessor = DataPreprocessor(apply_filtering=False)
    preprocessor.fit(trajectories)

    # Zapisanie statystyk preprocessora
    preprocessor.save_stats(str(output_dir / 'preprocessor_stats.npz'))

    # DataModule
    print("\nTworzenie DataModule...")
    datamodule = create_datamodule(trajectories, args, preprocessor)
    datamodule.setup()

    info = datamodule.get_data_info()
    print(f"  - Trajektorie: {info['num_trajectories']}")
    print(f"  - Próbki treningowe: {info['train_samples']}")
    print(f"  - Próbki walidacyjne: {info['val_samples']}")
    print(f"  - Próbki testowe: {info['test_samples']}")

    # Model
    print("\nTworzenie modelu...")
    if args.checkpoint:
        print(f"  Wznowienie z checkpointu: {args.checkpoint}")
        model = Seq2SeqModel.load_from_checkpoint(args.checkpoint)
    else:
        model = create_model(args)

    params = model.get_num_parameters()
    print(f"  - Parametry: {params['total']:,} (trenowalnych: {params['trainable']:,})")

    # PyTorch Lightning trainer
    trainer = create_trainer(args, output_dir)

    # Trening
    print("\nRozpoczęcie treningu...")
    print("-" * 40)

    trainer.fit(model, datamodule)

    print("-" * 40)
    print("Trening zakończony!")

    # Test na zbiorze testowym
    print("\nEwaluacja na zbiorze testowym...")
    test_results = trainer.test(model, datamodule)

    # Wizualizacja
    visualize_results(
        model=model,
        datamodule=datamodule,
        preprocessor=preprocessor,
        output_dir=output_dir,
        dataset=dataset
    )

    # Zapis wyników
    print(f"\nWyniki zapisane w: {output_dir}")


def test(args: argparse.Namespace) -> None:
    """
    Przeprowadza ewaluację modelu na zbiorze testowym.

    Args:
        args: Argumenty wiersza poleceń
    """
    print("\n" + "=" * 60)
    print("EWALUACJA MODELU")
    print("=" * 60)

    if args.checkpoint is None:
        print("BŁĄD: Wymagany jest argument --checkpoint dla trybu test")
        sys.exit(1)

    # Ustawienie ziarna
    setup_seed(args.seed)

    # Tworzenie katalogu wyjściowego
    output_dir = Path(args.output_dir) / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ładowanie danych
    dataset = load_or_generate_data(args)
    trajectories = dataset['trajectories']

    # Preprocessor
    checkpoint_dir = Path(args.checkpoint).parent.parent
    preprocessor_path = checkpoint_dir / 'preprocessor_stats.npz'

    preprocessor = DataPreprocessor()
    if preprocessor_path.exists():
        print(f"\nŁadowanie preprocessora z: {preprocessor_path}")
        preprocessor.load_stats(str(preprocessor_path))
    else:
        print("\nDopasowywanie nowego preprocessora...")
        preprocessor.fit(trajectories)

    # DataModule
    datamodule = create_datamodule(trajectories, args, preprocessor)
    datamodule.setup('test')

    # Model
    print(f"\nŁadowanie modelu z: {args.checkpoint}")
    model = Seq2SeqModel.load_from_checkpoint(args.checkpoint)
    model.eval()

    # Ewaluacja
    accelerator = get_accelerator(args.device)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False
    )

    print("\nEwaluacja na zbiorze testowym...")
    test_results = trainer.test(model, datamodule)

    # Szczegółowa ewaluacja
    print("\nObliczanie szczegółowych metryk...")
    evaluate_model(model, datamodule, preprocessor, output_dir)

    # Wizualizacja
    visualize_results(
        model=model,
        datamodule=datamodule,
        preprocessor=preprocessor,
        output_dir=output_dir,
        dataset=dataset
    )

    print(f"\nWyniki zapisane w: {output_dir}")


def evaluate_model(
    model: Seq2SeqModel,
    datamodule: TimeSeriesDataModule,
    preprocessor: DataPreprocessor,
    output_dir: Path
) -> Dict[str, float]:
    """
    Przeprowadza szczegółową ewaluację modelu.

    Args:
        model: Wytrenowany model
        datamodule: DataModule z danymi
        preprocessor: Preprocessor do denormalizacji
        output_dir: Katalog wyjściowy

    Returns:
        Słownik z metrykami
    """
    model.eval()
    device = next(model.parameters()).device

    all_mu = []
    all_sigma = []
    all_targets = []

    # Predykcja na całym zbiorze testowym
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            input_seq, target_seq = batch
            input_seq = input_seq.to(device)

            mu_seq, sigma_seq = model(input_seq, teacher_forcing_ratio=0.0)

            all_mu.append(mu_seq.cpu())
            all_sigma.append(sigma_seq.cpu())
            all_targets.append(target_seq)

    # Konkatenacja wyników
    mu = torch.cat(all_mu, dim=0).numpy()
    sigma = torch.cat(all_sigma, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # Obliczenie metryk
    metrics = compute_all_metrics(mu, sigma, targets, feature_names=['x', 'v'])

    # Wyświetlenie wyników
    print("\n" + "-" * 40)
    print("METRYKI EWALUACJI:")
    print("-" * 40)
    print(f"  RMSE:     {metrics['rmse']:.6f}")
    print(f"  MAE:      {metrics['mae']:.6f}")
    print(f"  NLL:      {metrics['nll']:.6f}")
    print(f"  CRPS:     {metrics['crps']:.6f}")
    print()
    print(f"  RMSE (x): {metrics['rmse_x']:.6f}")
    print(f"  RMSE (v): {metrics['rmse_v']:.6f}")
    print()
    print(f"  Coverage 1σ: {metrics['coverage_1.0sigma']:.2%} (oczekiwane: 68.27%)")
    print(f"  Coverage 2σ: {metrics['coverage_2.0sigma']:.2%} (oczekiwane: 95.45%)")
    print("-" * 40)

    # Zapis metryk do pliku
    metrics_path = output_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("METRYKI EWALUACJI\n")
        f.write("=" * 40 + "\n\n")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")

    print(f"\nMetryki zapisane do: {metrics_path}")

    return metrics


def visualize_results(
    model: Seq2SeqModel,
    datamodule: TimeSeriesDataModule,
    preprocessor: DataPreprocessor,
    output_dir: Path,
    dataset: Dict[str, np.ndarray]
) -> None:
    """
    Generuje wykresy wyników.

    Args:
        model: Wytrenowany model
        datamodule: DataModule z danymi
        preprocessor: Preprocessor do denormalizacji
        output_dir: Katalog wyjściowy
        dataset: Oryginalny dataset (dla wektora czasu)
    """
    print("\nGenerowanie wizualizacji...")

    # Katalog na wykresy
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    dt = dataset['time'][1] - dataset['time'][0]

    # Pobranie przykładowego batcha
    datamodule.setup('test')
    sample_batch = next(iter(datamodule.test_dataloader()))
    input_seq, target_seq = sample_batch

    # Predykcja
    with torch.no_grad():
        input_seq_device = input_seq.to(device)
        mu_seq, sigma_seq = model(input_seq_device, teacher_forcing_ratio=0.0)
        mu_seq = mu_seq.cpu()
        sigma_seq = sigma_seq.cpu()

    # Wybór przykładowej trajektorii
    idx = 0
    input_np = input_seq[idx].numpy()
    target_np = target_seq[idx].numpy()
    mu_np = mu_seq[idx].numpy()
    sigma_np = sigma_seq[idx].numpy()

    # Denormalizacja
    input_denorm = preprocessor.inverse_transform(input_np)
    target_denorm = preprocessor.inverse_transform(target_np)
    mu_denorm, sigma_denorm = preprocessor.inverse_transform_gaussian(mu_np, sigma_np)

    # Wektory czasu
    T_in = input_np.shape[0]
    T_out = target_np.shape[0]
    time_input = np.arange(0, T_in * dt, dt)
    time_output = np.arange(T_in * dt, (T_in + T_out) * dt, dt)

    import matplotlib.pyplot as plt

    # Wybór losowej pełnej trajektorii z datasetu do wizualizacji
    traj_idx = np.random.randint(0, dataset['trajectories'].shape[0])
    full_trajectory = dataset['trajectories'][traj_idx]  # (num_steps, 2)
    full_time = dataset['time']

    # Trajektoria z datasetu jest już w oryginalnych jednostkach (nie znormalizowana)
    full_trajectory_denorm = full_trajectory

    # Losowy punkt startowy dla okna predykcji
    max_start = len(full_time) - T_in - T_out
    input_start_idx = np.random.randint(0, max(1, max_start))

    # Przygotowanie danych do predykcji na pełnej trajektorii
    input_window = full_trajectory[input_start_idx:input_start_idx + T_in]
    input_window_norm = preprocessor.transform(input_window)
    input_tensor = torch.FloatTensor(input_window_norm).unsqueeze(0).to(device)

    # Predykcja dla pełnej trajektorii
    with torch.no_grad():
        mu_full, sigma_full = model(input_tensor, teacher_forcing_ratio=0.0)
        mu_full = mu_full.cpu().numpy()[0]
        sigma_full = sigma_full.cpu().numpy()[0]

    # Denormalizacja predykcji
    mu_full_denorm, sigma_full_denorm = preprocessor.inverse_transform_gaussian(mu_full, sigma_full)

    # 1. Pełna trajektoria - położenie
    print("  - Wykres pełnej trajektorii (położenie)...")
    fig1 = plot_full_trajectory_with_prediction(
        full_trajectory=full_trajectory_denorm,
        full_time=full_time,
        input_start_idx=input_start_idx,
        T_in=T_in,
        T_out=T_out,
        mu_seq=mu_full_denorm,
        sigma_seq=sigma_full_denorm,
        feature_idx=0,
        feature_name='Położenie x [m]',
        title=f'Pełna trajektoria #{traj_idx} z predykcją położenia',
        save_path=str(plots_dir / 'full_trajectory_position.png')
    )
    plt.close(fig1)

    # 2. Pełna trajektoria - prędkość
    print("  - Wykres pełnej trajektorii (prędkość)...")
    fig2 = plot_full_trajectory_with_prediction(
        full_trajectory=full_trajectory_denorm,
        full_time=full_time,
        input_start_idx=input_start_idx,
        T_in=T_in,
        T_out=T_out,
        mu_seq=mu_full_denorm,
        sigma_seq=sigma_full_denorm,
        feature_idx=1,
        feature_name='Prędkość v [m/s]',
        title=f'Pełna trajektoria #{traj_idx} z predykcją prędkości',
        save_path=str(plots_dir / 'full_trajectory_velocity.png')
    )
    plt.close(fig2)

    # 3. Predykcja położenia z niepewnością (zoom na fragment)
    print("  - Wykres predykcji położenia (zoom)...")
    fig3 = plot_prediction_with_uncertainty(
        time_input=time_input,
        time_output=time_output,
        input_seq=input_denorm,
        target_seq=target_denorm,
        mu_seq=mu_denorm,
        sigma_seq=sigma_denorm,
        feature_idx=0,
        feature_name='Położenie x [m]',
        title='Predykcja położenia z przedziałami ufności (zoom)',
        save_path=str(plots_dir / 'prediction_position_zoom.png')
    )
    plt.close(fig3)

    # 4. Predykcja prędkości z niepewnością (zoom na fragment)
    print("  - Wykres predykcji prędkości (zoom)...")
    fig4 = plot_prediction_with_uncertainty(
        time_input=time_input,
        time_output=time_output,
        input_seq=input_denorm,
        target_seq=target_denorm,
        mu_seq=mu_denorm,
        sigma_seq=sigma_denorm,
        feature_idx=1,
        feature_name='Prędkość v [m/s]',
        title='Predykcja prędkości z przedziałami ufności (zoom)',
        save_path=str(plots_dir / 'prediction_velocity_zoom.png')
    )
    plt.close(fig4)

    # 5. Porównanie obu cech
    print("  - Wykres porównania cech...")
    fig5 = plot_prediction_comparison(
        time=time_output,
        target=target_denorm,
        mu=mu_denorm,
        sigma=sigma_denorm,
        title='Porównanie predykcji',
        save_path=str(plots_dir / 'prediction_comparison.png')
    )
    plt.close(fig5)

    # 6. Ewolucja niepewności
    print("  - Wykres ewolucji niepewności...")
    fig6 = plot_uncertainty_evolution(
        time=time_output,
        sigma=sigma_denorm,
        title='Ewolucja niepewności w czasie',
        save_path=str(plots_dir / 'uncertainty_evolution.png')
    )
    plt.close(fig6)

    # 7. Portret fazowy
    print("  - Portret fazowy...")
    full_input = np.vstack([input_denorm, target_denorm])
    full_pred = np.vstack([input_denorm, mu_denorm])

    fig7 = plot_phase_space(
        trajectories=[full_input, full_pred],
        labels=['Rzeczywista', 'Predykcja'],
        title='Portret fazowy - porównanie trajektorii',
        save_path=str(plots_dir / 'phase_space.png')
    )
    plt.close(fig7)

    # 8. Krzywe uczenia (jeśli dostępne logi)
    logs_dir = output_dir / 'logs'
    if logs_dir.exists():
        # Szukanie pliku metrics.csv
        metrics_files = list(logs_dir.glob('**/metrics.csv'))
        if metrics_files:
            print("  - Krzywe uczenia...")
            import pandas as pd
            df = pd.read_csv(metrics_files[0])

            if 'train_loss_epoch' in df.columns and 'val_loss' in df.columns:
                train_losses = df['train_loss_epoch'].dropna().tolist()
                val_losses = df['val_loss'].dropna().tolist()

                # Wyrównanie długości
                min_len = min(len(train_losses), len(val_losses))
                train_losses = train_losses[:min_len]
                val_losses = val_losses[:min_len]

                if len(train_losses) > 0:
                    fig8 = plot_training_curves(
                        train_losses=train_losses,
                        val_losses=val_losses,
                        title='Krzywe uczenia',
                        save_path=str(plots_dir / 'training_curves.png')
                    )
                    plt.close(fig8)

    print(f"\nWykresy zapisane w: {plots_dir}")


def predict(args: argparse.Namespace) -> None:
    """
    Przeprowadza predykcję na losowej trajektorii syntetycznej.

    Generuje losową trajektorię oscylatora tłumionego i wykonuje
    wiele predykcji w różnych punktach czasowych.

    Args:
        args: Argumenty wiersza poleceń
    """
    import matplotlib.pyplot as plt
    from src.data_generation.synthetic import DampedOscillator

    print("\n" + "=" * 60)
    print("PREDYKCJA")
    print("=" * 60)

    if args.checkpoint is None:
        print("BŁĄD: Wymagany jest argument --checkpoint dla trybu predict")
        sys.exit(1)

    # Ustawienie ziarna (losowego jeśli nie podano)
    if args.seed == 42:  # domyślna wartość - użyj losowego
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    setup_seed(seed)
    print(f"Seed: {seed}")

    # Tworzenie katalogu wyjściowego
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / 'predictions' / f'pred_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ładowanie modelu
    print(f"\nŁadowanie modelu z: {args.checkpoint}")
    device = get_device(args.device)
    model = Seq2SeqModel.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    # Pobranie parametrów z modelu
    T_in = model.hparams.get('T_in', args.T_in)
    T_out = model.hparams.get('T_out', args.T_out)
    print(f"Parametry modelu: T_in={T_in}, T_out={T_out}")

    # Ładowanie preprocessora
    preprocessor = DataPreprocessor()
    checkpoint_dir = Path(args.checkpoint).parent.parent
    preprocessor_path = checkpoint_dir / 'preprocessor_stats.npz'

    if preprocessor_path.exists():
        preprocessor.load_stats(str(preprocessor_path))
        print(f"Załadowano preprocessor z: {preprocessor_path}")
    else:
        print("UWAGA: Brak pliku preprocessora - używam domyślnej normalizacji")

    # Generowanie losowej trajektorii z losowymi parametrami
    print("\nGenerowanie losowej trajektorii...")

    # Losowe parametry oscylatora (w sensownych zakresach)
    mass = np.random.uniform(0.5, 2.0)
    damping = np.random.uniform(0.1, 0.8)
    stiffness = np.random.uniform(1.0, 5.0)
    x0 = np.random.uniform(-2.0, 2.0)
    v0 = np.random.uniform(-1.0, 1.0)

    oscillator_params = {
        'mass': round(mass, 2),
        'damping': round(damping, 2),
        'stiffness': round(stiffness, 2),
        'x0': round(x0, 2),
        'v0': round(v0, 2)
    }

    print(f"  Parametry oscylatora:")
    print(f"    - masa (m): {oscillator_params['mass']}")
    print(f"    - tłumienie (c): {oscillator_params['damping']}")
    print(f"    - sztywność (k): {oscillator_params['stiffness']}")
    print(f"    - x₀: {oscillator_params['x0']}")
    print(f"    - v₀: {oscillator_params['v0']}")

    oscillator = DampedOscillator(mass=mass, damping=damping, stiffness=stiffness)

    # Generowanie trajektorii - wystarczająco długiej dla wielu predykcji
    dt = args.dt
    t_max = args.t_max
    t = np.arange(0, t_max, dt)
    x, v = oscillator.generate_trajectory(x0=x0, v0=v0, t=t)
    trajectory = np.column_stack([x, v])

    print(f"  Długość trajektorii: {len(t)} kroków ({t_max}s przy dt={dt}s)")

    # Normalizacja trajektorii
    if preprocessor_path.exists():
        trajectory_norm = preprocessor.transform(trajectory)
    else:
        # Fallback - dopasowanie do trajektorii
        preprocessor.fit(trajectory[np.newaxis, :, :])
        trajectory_norm = preprocessor.transform(trajectory)

    # Wybór punktów startowych dla predykcji
    num_predictions = args.num_predictions
    max_start = len(t) - T_in - T_out

    if max_start <= 0:
        print(f"BŁĄD: Trajektoria zbyt krótka dla T_in={T_in}, T_out={T_out}")
        sys.exit(1)

    # Równomierne rozmieszczenie punktów predykcji
    start_indices = np.linspace(0, max_start, num_predictions, dtype=int)

    print(f"\nWykonywanie {num_predictions} predykcji...")

    predictions_list = []
    for i, start_idx in enumerate(start_indices):
        # Przygotowanie danych wejściowych
        input_seq = trajectory_norm[start_idx:start_idx + T_in]
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)

        # Predykcja
        with torch.no_grad():
            pred = model.predict_trajectory(input_tensor, num_samples=50)

        mu = pred['mu'].cpu().numpy()
        sigma = pred['sigma'].cpu().numpy()

        # Denormalizacja
        mu_denorm, sigma_denorm = preprocessor.inverse_transform_gaussian(mu, sigma)

        predictions_list.append({
            'start_idx': start_idx,
            'mu': mu_denorm,
            'sigma': sigma_denorm
        })

        print(f"  Predykcja {i+1}/{num_predictions}: t_start={t[start_idx]:.2f}s")

    # Wizualizacja - pełne wykresy
    print("\nGenerowanie wizualizacji...")

    fig1 = plot_multi_prediction_trajectory(
        full_trajectory=trajectory,
        full_time=t,
        predictions=predictions_list,
        T_in=T_in,
        T_out=T_out,
        feature_idx=0,
        feature_name='Położenie x [m]',
        title=f'Predykcje położenia ({num_predictions} punktów)',
        save_path=str(output_dir / 'multi_prediction_position.png'),
        oscillator_params=oscillator_params
    )
    plt.close(fig1)

    fig2 = plot_multi_prediction_trajectory(
        full_trajectory=trajectory,
        full_time=t,
        predictions=predictions_list,
        T_in=T_in,
        T_out=T_out,
        feature_idx=1,
        feature_name='Prędkość v [m/s]',
        title=f'Predykcje prędkości ({num_predictions} punktów)',
        save_path=str(output_dir / 'multi_prediction_velocity.png'),
        oscillator_params=oscillator_params
    )
    plt.close(fig2)

    # Zoomy dla poszczególnych predykcji
    zooms_dir = output_dir / 'zooms'
    zooms_dir.mkdir(exist_ok=True)
    print(f"  Generowanie {num_predictions} zoomów...")

    for i, pred in enumerate(predictions_list):
        start_idx = pred['start_idx']
        mu = pred['mu']
        sigma = pred['sigma']

        # Wektory czasu dla tej predykcji
        time_input = t[start_idx:start_idx + T_in]
        time_output = t[start_idx + T_in:start_idx + T_in + T_out]

        # Dane wejściowe i docelowe
        input_seq = trajectory[start_idx:start_idx + T_in]
        target_seq = trajectory[start_idx + T_in:start_idx + T_in + T_out]

        # Zoom - położenie
        fig_zoom_x = plot_prediction_with_uncertainty(
            time_input=time_input,
            time_output=time_output,
            input_seq=input_seq,
            target_seq=target_seq,
            mu_seq=mu,
            sigma_seq=sigma,
            feature_idx=0,
            feature_name='Położenie x [m]',
            title=f'Predykcja #{i+1} - położenie (t={t[start_idx]:.1f}s)',
            save_path=str(zooms_dir / f'zoom_{i+1}_position.png')
        )
        plt.close(fig_zoom_x)

        # Zoom - prędkość
        fig_zoom_v = plot_prediction_with_uncertainty(
            time_input=time_input,
            time_output=time_output,
            input_seq=input_seq,
            target_seq=target_seq,
            mu_seq=mu,
            sigma_seq=sigma,
            feature_idx=1,
            feature_name='Prędkość v [m/s]',
            title=f'Predykcja #{i+1} - prędkość (t={t[start_idx]:.1f}s)',
            save_path=str(zooms_dir / f'zoom_{i+1}_velocity.png')
        )
        plt.close(fig_zoom_v)

    # Predykcja rekurencyjna (jeśli włączona)
    recursive_steps = args.recursive_steps
    if recursive_steps > 0:
        print(f"\n  Predykcja rekurencyjna ({recursive_steps} kroków)...")

        recursive_dir = output_dir / 'recursive'
        recursive_dir.mkdir(exist_ok=True)

        # Początkowe okno wejściowe (pierwsze T_in kroków)
        initial_input = trajectory[:T_in]
        initial_input_norm = trajectory_norm[:T_in]

        recursive_predictions = []
        current_input_norm = initial_input_norm.copy()

        for step in range(recursive_steps):
            # Predykcja
            input_tensor = torch.FloatTensor(current_input_norm).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model.predict_trajectory(input_tensor, num_samples=50)

            mu_norm = pred['mu'].cpu().numpy()
            sigma_norm = pred['sigma'].cpu().numpy()

            # Denormalizacja
            mu_denorm, sigma_denorm = preprocessor.inverse_transform_gaussian(mu_norm, sigma_norm)

            recursive_predictions.append({
                'mu': mu_denorm,
                'sigma': sigma_denorm,
                'mu_norm': mu_norm,
                'time_start': (T_in + step * T_out) * dt
            })

            print(f"    Krok {step+1}/{recursive_steps}: t={recursive_predictions[-1]['time_start']:.1f}s")

            # Przygotowanie wejścia dla następnego kroku
            # Używamy ostatnich T_in kroków z predykcji (lub całej jeśli T_out >= T_in)
            if T_out >= T_in:
                # Bierzemy ostatnie T_in kroków z predykcji
                current_input_norm = mu_norm[-T_in:]
            else:
                # Łączymy część starego wejścia z nową predykcją
                overlap = T_in - T_out
                current_input_norm = np.concatenate([
                    current_input_norm[-overlap:],
                    mu_norm
                ], axis=0)

        # Wizualizacja rekurencyjna - położenie
        fig_rec_x = plot_recursive_prediction(
            full_trajectory=trajectory,
            full_time=t,
            initial_input=initial_input,
            recursive_predictions=recursive_predictions,
            T_in=T_in,
            T_out=T_out,
            feature_idx=0,
            feature_name='Położenie x [m]',
            title=f'Predykcja rekurencyjna położenia ({recursive_steps} kroków)',
            save_path=str(recursive_dir / 'recursive_position.png'),
            oscillator_params=oscillator_params
        )
        plt.close(fig_rec_x)

        # Wizualizacja rekurencyjna - prędkość
        fig_rec_v = plot_recursive_prediction(
            full_trajectory=trajectory,
            full_time=t,
            initial_input=initial_input,
            recursive_predictions=recursive_predictions,
            T_in=T_in,
            T_out=T_out,
            feature_idx=1,
            feature_name='Prędkość v [m/s]',
            title=f'Predykcja rekurencyjna prędkości ({recursive_steps} kroków)',
            save_path=str(recursive_dir / 'recursive_velocity.png'),
            oscillator_params=oscillator_params
        )
        plt.close(fig_rec_v)

    # Zapisanie parametrów do pliku
    params_file = output_dir / 'parameters.txt'
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write(f"Seed: {seed}\n")
        f.write(f"T_in: {T_in}\n")
        f.write(f"T_out: {T_out}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"t_max: {t_max}\n")
        f.write(f"num_predictions: {num_predictions}\n")
        f.write(f"recursive_steps: {recursive_steps}\n")
        f.write(f"\nParametry oscylatora:\n")
        for k, v in oscillator_params.items():
            f.write(f"  {k}: {v}\n")

    print(f"\nWyniki zapisane w: {output_dir}")


def main() -> None:
    """
    Główna funkcja programu.
    """
    # Optymalizacje GPU
    if torch.cuda.is_available():
        # TF32 dla Tensor Cores (RTX 20xx, 30xx, 40xx, A-series)
        torch.set_float32_matmul_precision('medium')
        # cuDNN benchmark - automatyczny wybór najszybszych algorytmów dla LSTM
        torch.backends.cudnn.benchmark = True

    # Parsowanie argumentów
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("PREDYKCJA SZEREGÓW CZASOWYCH - MODEL SEQ2SEQ LSTM")
    print("=" * 60)
    print(f"\nTryb: {args.mode.upper()}")
    print(f"Seed: {args.seed}")

    # Informacje o urządzeniu (obsługa NVIDIA/AMD/Apple Silicon)
    print_device_info()
    device = get_device(args.device)
    print(f"Wybrane urządzenie: {device}")

    # Wykonanie odpowiedniego trybu
    if args.mode == 'generate':
        generate_data(args)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'predict':
        predict(args)

    print("\n" + "=" * 60)
    print("ZAKOŃCZONO")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
