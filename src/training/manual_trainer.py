"""
Ręczna pętla treningowa dla DirectML (AMD/Intel na Windows).

PyTorch Lightning nie wspiera DirectML, więc ten moduł implementuje
klasyczną pętlę treningową w czystym PyTorch.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.training.losses import multistep_gaussian_nll_loss


class EarlyStopping:
    """
    Mechanizm early stopping - zatrzymuje trening gdy metryka przestaje się poprawiać.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """
        Args:
            patience: Liczba epok bez poprawy przed zatrzymaniem
            min_delta: Minimalna zmiana uznawana za poprawę
            mode: 'min' dla minimalizacji, 'max' dla maksymalizacji
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Sprawdza czy należy zatrzymać trening.

        Args:
            value: Aktualna wartość metryki

        Returns:
            True jeśli należy zatrzymać trening
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ManualTrainer:
    """
    Ręczna pętla treningowa dla modeli Seq2Seq.

    Używana głównie dla DirectML na Windows, gdzie PyTorch Lightning
    nie jest wspierany. Implementuje pełny cykl treningowy z:
    - Treningiem i walidacją
    - Early stopping
    - Zapisywaniem checkpointów
    - Logowaniem metryk
    - Gradient clipping
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        gradient_clip_val: float = 1.0,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[Path] = None,
        log_callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Inicjalizacja trenera.

        Args:
            model: Model do trenowania
            device: Urządzenie (cpu, cuda, mps, directml)
            learning_rate: Współczynnik uczenia
            gradient_clip_val: Maksymalna norma gradientu
            early_stopping_patience: Cierpliwość dla early stopping
            checkpoint_dir: Katalog na checkpointy
            log_callback: Opcjonalna funkcja do logowania metryk
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.gradient_clip_val = gradient_clip_val
        self.checkpoint_dir = checkpoint_dir
        self.log_callback = log_callback

        # Optymalizator
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='min'
        )

        # Historia treningu
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'learning_rate': [],
        }

        # Najlepszy model
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def _move_batch_to_device(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Przenosi batch na urządzenie."""
        inputs, targets = batch
        return inputs.to(self.device), targets.to(self.device)

    def _train_epoch(
        self,
        train_loader: DataLoader,
        teacher_forcing_ratio: float
    ) -> float:
        """
        Pojedyncza epoka treningowa.

        Args:
            train_loader: DataLoader z danymi treningowymi
            teacher_forcing_ratio: Współczynnik teacher forcing

        Returns:
            Średnia strata treningowa
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            inputs, targets = self._move_batch_to_device(batch)

            # Forward pass
            self.optimizer.zero_grad()
            mu_seq, sigma_seq = self.model(
                x=inputs,
                target=targets,
                teacher_forcing_ratio=teacher_forcing_ratio
            )

            # Obliczenie straty
            loss = multistep_gaussian_nll_loss(
                mu_seq=mu_seq,
                sigma_seq=sigma_seq,
                target_seq=targets,
                reduction='mean'
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )

            # Aktualizacja wag
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Walidacja modelu.

        Args:
            val_loader: DataLoader z danymi walidacyjnymi

        Returns:
            Tuple (strata walidacyjna, RMSE)
        """
        self.model.eval()
        total_loss = 0.0
        total_rmse = 0.0
        num_batches = 0

        for batch in val_loader:
            inputs, targets = self._move_batch_to_device(batch)

            # Forward pass (bez teacher forcing)
            mu_seq, sigma_seq = self.model(
                x=inputs,
                target=None,
                teacher_forcing_ratio=0.0
            )

            # Obliczenie straty
            loss = multistep_gaussian_nll_loss(
                mu_seq=mu_seq,
                sigma_seq=sigma_seq,
                target_seq=targets,
                reduction='mean'
            )

            # RMSE
            rmse = torch.sqrt(((mu_seq - targets) ** 2).mean())

            total_loss += loss.item()
            total_rmse += rmse.item()
            num_batches += 1

        return total_loss / num_batches, total_rmse / num_batches

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Zapisuje checkpoint modelu."""
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
        }

        # Zapisz ostatni checkpoint
        last_path = self.checkpoint_dir / 'last.ckpt'
        torch.save(checkpoint, last_path)

        # Zapisz najlepszy checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f'best_model-epoch={epoch:02d}-val_loss={val_loss:.4f}.ckpt'
            torch.save(checkpoint, best_path)
            # Usuń poprzednie najlepsze checkpointy
            for old_ckpt in self.checkpoint_dir.glob('best_model-*.ckpt'):
                if old_ckpt != best_path:
                    old_ckpt.unlink()

    def load_checkpoint(self, checkpoint_path: Path):
        """Ładuje checkpoint modelu."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint.get('epoch', 0)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        teacher_forcing_ratio: float = 1.0,
        teacher_forcing_decay: float = 0.02,
    ) -> Dict:
        """
        Trenuje model.

        Args:
            train_loader: DataLoader z danymi treningowymi
            val_loader: DataLoader z danymi walidacyjnymi
            max_epochs: Maksymalna liczba epok
            teacher_forcing_ratio: Początkowy współczynnik teacher forcing
            teacher_forcing_decay: Spadek TF na epokę

        Returns:
            Słownik z historią treningu
        """
        print("\n" + "=" * 60)
        print("TRENING (ręczna pętla - DirectML)")
        print("=" * 60)
        print(f"Urządzenie: {self.device}")
        print(f"Epoki: {max_epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Teacher forcing: {teacher_forcing_ratio} (decay: {teacher_forcing_decay})")
        print("=" * 60 + "\n")

        current_tf_ratio = teacher_forcing_ratio
        start_time = time.time()

        for epoch in range(max_epochs):
            epoch_start = time.time()

            # Trening
            train_loss = self._train_epoch(train_loader, current_tf_ratio)

            # Walidacja
            val_loss, val_rmse = self._validate_epoch(val_loader)

            # Aktualizacja schedulera
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Zapisanie historii
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_rmse'].append(val_rmse)
            self.history['learning_rate'].append(current_lr)

            # Sprawdzenie czy to najlepszy model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()

            # Zapisanie checkpointu
            self._save_checkpoint(epoch, val_loss, is_best)

            # Logowanie
            epoch_time = time.time() - epoch_start
            best_marker = " *" if is_best else ""
            print(
                f"Epoka {epoch + 1:3d}/{max_epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_rmse: {val_rmse:.4f} | "
                f"lr: {current_lr:.2e} | "
                f"TF: {current_tf_ratio:.2f} | "
                f"czas: {epoch_time:.1f}s{best_marker}"
            )

            if self.log_callback:
                self.log_callback({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'learning_rate': current_lr,
                    'teacher_forcing_ratio': current_tf_ratio,
                })

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping po {epoch + 1} epokach!")
                break

            # Redukcja teacher forcing
            current_tf_ratio = max(0.0, current_tf_ratio - teacher_forcing_decay)

        # Podsumowanie
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Trening zakończony w {total_time / 60:.1f} minut")
        print(f"Najlepsza val_loss: {self.best_val_loss:.4f}")
        print("=" * 60)

        # Przywróć najlepszy model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Testuje model na zbiorze testowym.

        Args:
            test_loader: DataLoader z danymi testowymi

        Returns:
            Słownik z metrykami testowymi
        """
        self.model.eval()
        total_loss = 0.0
        total_rmse = 0.0
        total_mae = 0.0
        num_batches = 0

        for batch in test_loader:
            inputs, targets = self._move_batch_to_device(batch)

            # Forward pass
            mu_seq, sigma_seq = self.model(
                x=inputs,
                target=None,
                teacher_forcing_ratio=0.0
            )

            # Metryki
            loss = multistep_gaussian_nll_loss(
                mu_seq=mu_seq,
                sigma_seq=sigma_seq,
                target_seq=targets,
                reduction='mean'
            )
            rmse = torch.sqrt(((mu_seq - targets) ** 2).mean())
            mae = (mu_seq - targets).abs().mean()

            total_loss += loss.item()
            total_rmse += rmse.item()
            total_mae += mae.item()
            num_batches += 1

        metrics = {
            'test_loss': total_loss / num_batches,
            'test_rmse': total_rmse / num_batches,
            'test_mae': total_mae / num_batches,
        }

        print("\n" + "-" * 40)
        print("WYNIKI TESTOWE:")
        print("-" * 40)
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        print("-" * 40)

        return metrics


def create_manual_trainer(
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
    learning_rate: float = 1e-3,
    gradient_clip_val: float = 1.0,
    early_stopping_patience: int = 10,
) -> ManualTrainer:
    """
    Tworzy ManualTrainer z domyślną konfiguracją.

    Args:
        model: Model do trenowania
        device: Urządzenie obliczeniowe
        output_dir: Katalog na wyniki
        learning_rate: Współczynnik uczenia
        gradient_clip_val: Maksymalna norma gradientu
        early_stopping_patience: Cierpliwość dla early stopping

    Returns:
        Skonfigurowany ManualTrainer
    """
    checkpoint_dir = output_dir / 'checkpoints'

    return ManualTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        gradient_clip_val=gradient_clip_val,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=checkpoint_dir,
    )
