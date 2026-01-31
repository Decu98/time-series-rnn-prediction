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
)
from src.training.manual_trainer import ManualTrainer, create_manual_trainer


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
        default=1.0,
        help='Początkowy współczynnik teacher forcing'
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

    return dataset


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


def should_use_manual_trainer(device_arg: str) -> bool:
    """
    Sprawdza czy należy użyć ręcznej pętli treningowej.

    DirectML nie jest wspierany przez PyTorch Lightning,
    więc dla DirectML zawsze używamy ManualTrainer.

    Args:
        device_arg: Argument --device z wiersza poleceń

    Returns:
        True jeśli należy użyć ManualTrainer
    """
    if device_arg == "directml":
        return True

    # Auto-detekcja: jeśli DirectML jest jedynym dostępnym GPU
    if device_arg == "auto":
        import platform
        if platform.system() == "Windows":
            if not torch.cuda.is_available() and is_directml_available():
                return True

    return False


def train_with_manual_trainer(
    args: argparse.Namespace,
    output_dir: Path,
    dataset: Dict[str, np.ndarray],
    preprocessor: DataPreprocessor,
    datamodule: TimeSeriesDataModule,
    model: Seq2SeqModel,
    device: torch.device
) -> None:
    """
    Trening z użyciem ręcznej pętli (dla DirectML).

    Args:
        args: Argumenty wiersza poleceń
        output_dir: Katalog wyjściowy
        dataset: Dane
        preprocessor: Preprocessor
        datamodule: DataModule
        model: Model do trenowania
        device: Urządzenie obliczeniowe
    """
    # Tworzenie ręcznego trenera
    trainer = create_manual_trainer(
        model=model,
        device=device,
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        gradient_clip_val=args.gradient_clip,
        early_stopping_patience=args.early_stopping_patience,
    )

    # Trening
    trainer.fit(
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.val_dataloader(),
        max_epochs=args.max_epochs,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        teacher_forcing_decay=0.02,
    )

    # Test
    print("\nEwaluacja na zbiorze testowym...")
    trainer.test(datamodule.test_dataloader())

    # Wizualizacja
    visualize_results(
        model=model,
        datamodule=datamodule,
        preprocessor=preprocessor,
        output_dir=output_dir,
        dataset=dataset
    )


def train(args: argparse.Namespace) -> None:
    """
    Przeprowadza trening modelu.

    Automatycznie wybiera między PyTorch Lightning (CUDA, MPS)
    a ręczną pętlą treningową (DirectML dla AMD na Windows).

    Args:
        args: Argumenty wiersza poleceń
    """
    print("\n" + "=" * 60)
    print("TRENING MODELU")
    print("=" * 60)

    # Ustawienie ziarna
    setup_seed(args.seed)

    # Sprawdzenie czy używamy DirectML
    use_manual = should_use_manual_trainer(args.device)
    if use_manual:
        print("\n[INFO] Wykryto DirectML - używam ręcznej pętli treningowej")

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

    # Wybór trenera
    if use_manual:
        # Ręczna pętla dla DirectML
        device = get_device(args.device)
        train_with_manual_trainer(
            args=args,
            output_dir=output_dir,
            dataset=dataset,
            preprocessor=preprocessor,
            datamodule=datamodule,
            model=model,
            device=device
        )
    else:
        # PyTorch Lightning dla CUDA/MPS/CPU
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

    # 1. Predykcja położenia z niepewnością
    print("  - Wykres predykcji położenia...")
    fig1 = plot_prediction_with_uncertainty(
        time_input=time_input,
        time_output=time_output,
        input_seq=input_denorm,
        target_seq=target_denorm,
        mu_seq=mu_denorm,
        sigma_seq=sigma_denorm,
        feature_idx=0,
        feature_name='Położenie x [m]',
        title='Predykcja położenia z przedziałami ufności',
        save_path=str(plots_dir / 'prediction_position.png')
    )
    import matplotlib.pyplot as plt
    plt.close(fig1)

    # 2. Predykcja prędkości z niepewnością
    print("  - Wykres predykcji prędkości...")
    fig2 = plot_prediction_with_uncertainty(
        time_input=time_input,
        time_output=time_output,
        input_seq=input_denorm,
        target_seq=target_denorm,
        mu_seq=mu_denorm,
        sigma_seq=sigma_denorm,
        feature_idx=1,
        feature_name='Prędkość v [m/s]',
        title='Predykcja prędkości z przedziałami ufności',
        save_path=str(plots_dir / 'prediction_velocity.png')
    )
    plt.close(fig2)

    # 3. Porównanie obu cech
    print("  - Wykres porównania cech...")
    fig3 = plot_prediction_comparison(
        time=time_output,
        target=target_denorm,
        mu=mu_denorm,
        sigma=sigma_denorm,
        title='Porównanie predykcji',
        save_path=str(plots_dir / 'prediction_comparison.png')
    )
    plt.close(fig3)

    # 4. Ewolucja niepewności
    print("  - Wykres ewolucji niepewności...")
    fig4 = plot_uncertainty_evolution(
        time=time_output,
        sigma=sigma_denorm,
        title='Ewolucja niepewności w czasie',
        save_path=str(plots_dir / 'uncertainty_evolution.png')
    )
    plt.close(fig4)

    # 5. Portret fazowy
    print("  - Portret fazowy...")
    full_input = np.vstack([input_denorm, target_denorm])
    full_pred = np.vstack([input_denorm, mu_denorm])

    fig5 = plot_phase_space(
        trajectories=[full_input, full_pred],
        labels=['Rzeczywista', 'Predykcja'],
        title='Portret fazowy - porównanie trajektorii',
        save_path=str(plots_dir / 'phase_space.png')
    )
    plt.close(fig5)

    # 6. Krzywe uczenia (jeśli dostępne logi)
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
                    fig6 = plot_training_curves(
                        train_losses=train_losses,
                        val_losses=val_losses,
                        title='Krzywe uczenia',
                        save_path=str(plots_dir / 'training_curves.png')
                    )
                    plt.close(fig6)

    print(f"\nWykresy zapisane w: {plots_dir}")


def predict(args: argparse.Namespace) -> None:
    """
    Przeprowadza predykcję dla pojedynczej trajektorii.

    Args:
        args: Argumenty wiersza poleceń
    """
    print("\n" + "=" * 60)
    print("PREDYKCJA")
    print("=" * 60)

    if args.checkpoint is None:
        print("BŁĄD: Wymagany jest argument --checkpoint dla trybu predict")
        sys.exit(1)

    # Ustawienie ziarna
    setup_seed(args.seed)

    # Tworzenie katalogu wyjściowego
    output_dir = Path(args.output_dir) / 'predictions'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ładowanie modelu
    print(f"\nŁadowanie modelu z: {args.checkpoint}")
    model = Seq2SeqModel.load_from_checkpoint(args.checkpoint)
    model.eval()

    # Generowanie pojedynczej trajektorii testowej
    print("\nGenerowanie trajektorii testowej...")
    from src.data_generation.synthetic import DampedOscillator

    oscillator = DampedOscillator(mass=1.0, damping=0.3, stiffness=2.0)
    t = np.arange(0, args.t_max, args.dt)
    x, v = oscillator.generate_trajectory(x0=1.5, v0=0.0, t=t)
    trajectory = np.column_stack([x, v])

    # Preprocessing
    preprocessor = DataPreprocessor()
    checkpoint_dir = Path(args.checkpoint).parent.parent
    preprocessor_path = checkpoint_dir / 'preprocessor_stats.npz'

    if preprocessor_path.exists():
        preprocessor.load_stats(str(preprocessor_path))
    else:
        # Fallback - dopasowanie do trajektorii
        preprocessor.fit(trajectory[np.newaxis, :, :])

    # Normalizacja
    trajectory_norm = preprocessor.transform(trajectory)

    # Wybranie okna wejściowego
    start_idx = 100  # Początek okna
    input_seq = trajectory_norm[start_idx:start_idx + args.T_in]
    target_seq = trajectory_norm[start_idx + args.T_in:start_idx + args.T_in + args.T_out]

    # Konwersja do tensora
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0)

    # Predykcja
    print("\nWykonywanie predykcji...")
    with torch.no_grad():
        predictions = model.predict_trajectory(input_tensor, num_samples=100)

    mu = predictions['mu'].numpy()
    sigma = predictions['sigma'].numpy()
    samples = predictions['samples'].numpy()

    # Denormalizacja
    input_denorm = preprocessor.inverse_transform(input_seq)
    target_denorm = preprocessor.inverse_transform(target_seq)
    mu_denorm, sigma_denorm = preprocessor.inverse_transform_gaussian(mu, sigma)

    # Wektory czasu
    time_input = t[start_idx:start_idx + args.T_in]
    time_output = t[start_idx + args.T_in:start_idx + args.T_in + args.T_out]

    # Wizualizacja
    print("\nGenerowanie wizualizacji...")

    fig = plot_prediction_with_uncertainty(
        time_input=time_input,
        time_output=time_output,
        input_seq=input_denorm,
        target_seq=target_denorm,
        mu_seq=mu_denorm,
        sigma_seq=sigma_denorm,
        feature_idx=0,
        feature_name='Położenie x [m]',
        title='Predykcja położenia oscylatora tłumionego',
        save_path=str(output_dir / 'single_prediction.png')
    )

    import matplotlib.pyplot as plt
    plt.close(fig)

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
