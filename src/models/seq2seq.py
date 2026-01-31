"""
Model Seq2Seq jako LightningModule.

Łączy Encoder i Decoder w pełny model predykcji szeregów czasowych
z integracją PyTorch Lightning.
"""

from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .decoder import GaussianDecoder
from .encoder import LSTMEncoder
from src.training.losses import multistep_gaussian_nll_loss


class Seq2SeqModel(pl.LightningModule):
    """
    Model Encoder-Decoder (Seq2Seq) dla predykcji szeregów czasowych.

    Model składa się z:
    - LSTMEncoder: koduje sekwencję wejściową do reprezentacji kontekstowej
    - GaussianDecoder: generuje sekwencję predykcji jako rozkłady Gaussowskie

    Zintegrowany z PyTorch Lightning dla łatwego treningu i ewaluacji.

    Attributes:
        encoder: Moduł encodera LSTM
        decoder: Moduł decodera Gaussowskiego
        T_out: Długość horyzontu predykcji
        teacher_forcing_ratio: Aktualny współczynnik teacher forcing
        learning_rate: Współczynnik uczenia
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        T_out: int = 50,
        dropout: float = 0.1,
        bidirectional_encoder: bool = False,
        learning_rate: float = 1e-3,
        teacher_forcing_ratio: float = 1.0,
        teacher_forcing_decay: float = 0.02,
        min_sigma: float = 1e-4,
        gradient_clip_val: float = 1.0
    ):
        """
        Inicjalizacja modelu Seq2Seq.

        Args:
            input_size: Liczba cech wejściowych (np. 2 dla [x, v])
            hidden_size: Rozmiar warstwy ukrytej LSTM
            num_layers: Liczba warstw LSTM
            T_out: Długość horyzontu predykcji
            dropout: Współczynnik dropout
            bidirectional_encoder: Czy encoder ma być dwukierunkowy
            learning_rate: Współczynnik uczenia dla optymalizatora
            teacher_forcing_ratio: Początkowy współczynnik teacher forcing (0-1)
            teacher_forcing_decay: Spadek teacher forcing na epokę
            min_sigma: Minimalna wartość sigma w decoderze
            gradient_clip_val: Maksymalna norma gradientu
        """
        super().__init__()

        # Zapisanie hiperparametrów (Lightning automatycznie je śledzi)
        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.T_out = T_out
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_decay = teacher_forcing_decay
        self.gradient_clip_val = gradient_clip_val

        # Encoder
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional_encoder
        )

        # Decoder
        self.decoder = GaussianDecoder(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=input_size,
            num_layers=num_layers,
            dropout=dropout,
            min_sigma=min_sigma
        )

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass modelu.

        Args:
            x: Sekwencja wejściowa (batch_size, T_in, input_size)
            target: Opcjonalna sekwencja docelowa dla teacher forcing
                   (batch_size, T_out, input_size)
            teacher_forcing_ratio: Opcjonalny współczynnik TF (nadpisuje domyślny)

        Returns:
            Tuple zawierający:
                - mu_seq: Predykowane średnie (batch_size, T_out, input_size)
                - sigma_seq: Predykowane odchylenia std (batch_size, T_out, input_size)
        """
        # Encoding
        _, encoder_hidden = self.encoder(x)

        # Ostatni krok z wejścia jako początek dla decodera
        initial_input = x[:, -1, :]  # (batch_size, input_size)

        # Wybór współczynnika teacher forcing
        tf_ratio = teacher_forcing_ratio if teacher_forcing_ratio is not None \
            else self.teacher_forcing_ratio

        # Decoding
        mu_seq, sigma_seq = self.decoder.forward_sequence(
            initial_input=initial_input,
            hidden=encoder_hidden,
            seq_len=self.T_out,
            target=target,
            teacher_forcing_ratio=tf_ratio
        )

        return mu_seq, sigma_seq

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Pojedynczy krok treningowy.

        Args:
            batch: Tuple (input_window, target_window)
            batch_idx: Indeks batcha

        Returns:
            Strata treningowa
        """
        input_seq, target_seq = batch

        # Forward pass z teacher forcing
        mu_seq, sigma_seq = self(
            x=input_seq,
            target=target_seq,
            teacher_forcing_ratio=self.teacher_forcing_ratio
        )

        # Obliczenie straty (Gaussian NLL)
        loss = multistep_gaussian_nll_loss(
            mu_seq=mu_seq,
            sigma_seq=sigma_seq,
            target_seq=target_seq,
            reduction='mean'
        )

        # Logowanie metryk
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('teacher_forcing_ratio', self.teacher_forcing_ratio, on_step=False, on_epoch=True)

        # Średnie sigma (do monitorowania)
        self.log('train_sigma_mean', sigma_seq.mean(), on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Pojedynczy krok walidacyjny.

        Args:
            batch: Tuple (input_window, target_window)
            batch_idx: Indeks batcha

        Returns:
            Strata walidacyjna
        """
        input_seq, target_seq = batch

        # Forward pass BEZ teacher forcing (pełna autoregresja)
        mu_seq, sigma_seq = self(
            x=input_seq,
            target=None,
            teacher_forcing_ratio=0.0
        )

        # Obliczenie straty
        loss = multistep_gaussian_nll_loss(
            mu_seq=mu_seq,
            sigma_seq=sigma_seq,
            target_seq=target_seq,
            reduction='mean'
        )

        # Obliczenie dodatkowych metryk
        rmse = torch.sqrt(((mu_seq - target_seq) ** 2).mean())
        mae = (mu_seq - target_seq).abs().mean()

        # Logowanie metryk
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_rmse', rmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        self.log('val_sigma_mean', sigma_seq.mean(), on_step=False, on_epoch=True)

        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Pojedynczy krok testowy.

        Args:
            batch: Tuple (input_window, target_window)
            batch_idx: Indeks batcha

        Returns:
            Strata testowa
        """
        input_seq, target_seq = batch

        # Forward pass BEZ teacher forcing
        mu_seq, sigma_seq = self(
            x=input_seq,
            target=None,
            teacher_forcing_ratio=0.0
        )

        # Obliczenie straty
        loss = multistep_gaussian_nll_loss(
            mu_seq=mu_seq,
            sigma_seq=sigma_seq,
            target_seq=target_seq,
            reduction='mean'
        )

        # Metryki
        rmse = torch.sqrt(((mu_seq - target_seq) ** 2).mean())
        mae = (mu_seq - target_seq).abs().mean()

        # Logowanie
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_rmse', rmse, on_step=False, on_epoch=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True)
        self.log('test_sigma_mean', sigma_seq.mean(), on_step=False, on_epoch=True)

        return loss

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Krok predykcji.

        Args:
            batch: Tuple (input_window, target_window)
            batch_idx: Indeks batcha

        Returns:
            Słownik z predykcjami i targetami
        """
        input_seq, target_seq = batch

        # Forward pass
        mu_seq, sigma_seq = self(
            x=input_seq,
            target=None,
            teacher_forcing_ratio=0.0
        )

        return {
            'input': input_seq,
            'target': target_seq,
            'mu': mu_seq,
            'sigma': sigma_seq
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Konfiguracja optymalizatora i schedulera.

        Returns:
            Słownik z konfiguracją optymalizatora
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def on_train_epoch_end(self) -> None:
        """
        Wywoływane na końcu każdej epoki treningowej.

        Aktualizuje współczynnik teacher forcing (scheduled sampling).
        """
        # Redukcja teacher forcing
        self.teacher_forcing_ratio = max(
            0.0,
            self.teacher_forcing_ratio - self.teacher_forcing_decay
        )

    def predict_trajectory(
        self,
        input_seq: torch.Tensor,
        num_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Predykcja trajektorii z wielokrotnym próbkowaniem.

        Generuje wiele próbek z rozkładu predykcji dla
        oszacowania niepewności.

        Args:
            input_seq: Sekwencja wejściowa (1, T_in, input_size) lub (T_in, input_size)
            num_samples: Liczba próbek do wygenerowania

        Returns:
            Słownik zawierający:
                - mu: Średnie predykcji (T_out, input_size)
                - sigma: Odchylenia std (T_out, input_size)
                - samples: Próbki z rozkładu (num_samples, T_out, input_size)
        """
        self.eval()

        # Upewnienie się o poprawnym kształcie
        if input_seq.dim() == 2:
            input_seq = input_seq.unsqueeze(0)

        with torch.no_grad():
            # Predykcja parametrów rozkładu
            mu_seq, sigma_seq = self(input_seq, teacher_forcing_ratio=0.0)

            # Próbkowanie
            samples = []
            for _ in range(num_samples):
                sample = self.decoder.sample(mu_seq, sigma_seq)
                samples.append(sample)

            samples = torch.cat(samples, dim=0)  # (num_samples, T_out, features)

        return {
            'mu': mu_seq.squeeze(0),  # (T_out, features)
            'sigma': sigma_seq.squeeze(0),  # (T_out, features)
            'samples': samples  # (num_samples, T_out, features)
        }

    def get_num_parameters(self) -> Dict[str, int]:
        """
        Zwraca liczbę parametrów modelu.

        Returns:
            Słownik z liczbą parametrów
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        return {
            'total': total,
            'trainable': trainable,
            'encoder': encoder_params,
            'decoder': decoder_params
        }


if __name__ == "__main__":
    # Test modułu Seq2Seq
    print("Test Seq2SeqModel (LightningModule)")
    print("=" * 60)

    # Parametry
    batch_size = 32
    T_in = 50
    T_out = 30
    input_size = 2
    hidden_size = 64
    num_layers = 2

    # Tworzenie modelu
    model = Seq2SeqModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        T_out=T_out,
        dropout=0.1,
        learning_rate=1e-3,
        teacher_forcing_ratio=0.5
    )

    print(f"\nArchitektura modelu:")
    print(model)

    # Informacje o parametrach
    params_info = model.get_num_parameters()
    print(f"\nLiczba parametrów:")
    for key, value in params_info.items():
        print(f"  {key}: {value:,}")

    # Przygotowanie danych testowych
    input_seq = torch.randn(batch_size, T_in, input_size)
    target_seq = torch.randn(batch_size, T_out, input_size)

    # Test forward pass
    print("\n" + "-" * 40)
    print("Test forward pass:")

    mu_seq, sigma_seq = model(input_seq, target_seq, teacher_forcing_ratio=0.5)
    print(f"Input shape: {input_seq.shape}")
    print(f"Output mu shape: {mu_seq.shape}")
    print(f"Output sigma shape: {sigma_seq.shape}")
    print(f"sigma min: {sigma_seq.min().item():.6f}")

    # Test training step
    print("\n" + "-" * 40)
    print("Test training_step:")

    batch = (input_seq, target_seq)
    loss = model.training_step(batch, 0)
    print(f"Training loss: {loss.item():.4f}")

    # Test validation step
    print("\n" + "-" * 40)
    print("Test validation_step:")

    val_loss = model.validation_step(batch, 0)
    print(f"Validation loss: {val_loss.item():.4f}")

    # Test predict_trajectory
    print("\n" + "-" * 40)
    print("Test predict_trajectory:")

    single_input = input_seq[0]  # (T_in, features)
    predictions = model.predict_trajectory(single_input, num_samples=10)

    print(f"mu shape: {predictions['mu'].shape}")
    print(f"sigma shape: {predictions['sigma'].shape}")
    print(f"samples shape: {predictions['samples'].shape}")

    # Test gradientów
    print("\n" + "-" * 40)
    print("Test gradientów:")

    model.zero_grad()
    loss = model.training_step(batch, 0)
    loss.backward()

    grad_exists = all(
        p.grad is not None for p in model.parameters()
        if p.requires_grad
    )
    print(f"Gradienty obliczone: {grad_exists}")

    # Test konfiguracji optymalizatora
    print("\n" + "-" * 40)
    print("Test configure_optimizers:")

    opt_config = model.configure_optimizers()
    print(f"Optimizer: {type(opt_config['optimizer']).__name__}")
    print(f"Scheduler: {type(opt_config['lr_scheduler']['scheduler']).__name__}")

    print("\n" + "=" * 60)
    print("Test zakończony pomyślnie!")
