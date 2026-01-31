"""
Moduł decodera z wyjściem Gaussowskim.

Decoder generuje sekwencję predykcji w postaci parametrów
rozkładu Gaussowskiego (μ, σ) dla każdego kroku czasowego.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class GaussianDecoder(nn.Module):
    """
    Decoder LSTM z wyjściem probabilistycznym (rozkład Gaussa).

    Generuje parametry rozkładu normalnego dla każdego kroku
    predykcji: średnią μ i odchylenie standardowe σ.

    Warstwa wyjściowa produkuje 2*num_features wartości:
    - pierwsza połowa: μ (średnie)
    - druga połowa: pre-σ (przed aktywacją softplus)

    Attributes:
        input_size: Liczba cech wejściowych
        hidden_size: Rozmiar warstwy ukrytej
        output_size: Liczba cech wyjściowych
        num_layers: Liczba warstw LSTM
        min_sigma: Minimalna wartość σ (stabilność numeryczna)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        min_sigma: float = 1e-4
    ):
        """
        Inicjalizacja decodera.

        Args:
            input_size: Liczba cech wejściowych (= output_size dla autoregresji)
            hidden_size: Rozmiar warstwy ukrytej LSTM
            output_size: Liczba cech wyjściowych (np. 2 dla [x, v])
            num_layers: Liczba warstw LSTM
            dropout: Współczynnik dropout
            min_sigma: Minimalna wartość σ dla stabilności
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.min_sigma = min_sigma

        # Warstwa LSTM decodera
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Warstwa wyjściowa - produkuje μ i pre-σ
        # Wyjście: 2 * output_size (dla μ i σ)
        self.output_layer = nn.Linear(hidden_size, 2 * output_size)

        # Aktywacja dla σ (zapewnia σ > 0)
        self.softplus = nn.Softplus()

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pojedynczy krok decodera (autoregresyjny).

        Args:
            x: Tensor wejściowy (batch_size, 1, input_size)
            hidden: Stan ukryty z poprzedniego kroku (h, c)

        Returns:
            Tuple zawierający:
                - mu: Średnie rozkładu (batch_size, output_size)
                - sigma: Odchylenia standardowe (batch_size, output_size)
                - (h_n, c_n): Nowe stany ukryte
        """
        # Przepuszczenie przez LSTM
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)

        # lstm_out: (batch_size, 1, hidden_size)
        # Usuwamy wymiar sekwencji
        lstm_out = lstm_out.squeeze(1)  # (batch_size, hidden_size)

        # Layer Normalization
        lstm_out = self.layer_norm(lstm_out)

        # Warstwa wyjściowa
        output = self.output_layer(lstm_out)  # (batch_size, 2 * output_size)

        # Rozdzielenie na μ i σ
        mu = output[:, :self.output_size]  # (batch_size, output_size)
        pre_sigma = output[:, self.output_size:]  # (batch_size, output_size)

        # Aktywacja softplus dla σ + minimalna wartość
        sigma = self.softplus(pre_sigma) + self.min_sigma

        return mu, sigma, (h_n, c_n)

    def forward_sequence(
        self,
        initial_input: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        seq_len: int,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generuje całą sekwencję predykcji.

        Może używać teacher forcing (podawanie prawdziwych wartości
        zamiast predykcji) lub pełnej autoregresji.

        Args:
            initial_input: Ostatni krok z okna wejściowego (batch_size, input_size)
            hidden: Stan ukryty z encodera (h, c)
            seq_len: Długość sekwencji do wygenerowania
            target: Opcjonalna sekwencja docelowa dla teacher forcing
                   (batch_size, seq_len, output_size)
            teacher_forcing_ratio: Prawdopodobieństwo użycia teacher forcing

        Returns:
            Tuple zawierający:
                - mu_seq: Sekwencja średnich (batch_size, seq_len, output_size)
                - sigma_seq: Sekwencja odchyleń std (batch_size, seq_len, output_size)
        """
        batch_size = initial_input.size(0)
        device = initial_input.device

        # Inicjalizacja wyjść
        mu_seq = torch.zeros(batch_size, seq_len, self.output_size, device=device)
        sigma_seq = torch.zeros(batch_size, seq_len, self.output_size, device=device)

        # Pierwszy input to ostatni krok z okna wejściowego
        decoder_input = initial_input.unsqueeze(1)  # (batch_size, 1, input_size)

        for t in range(seq_len):
            # Krok decodera
            mu, sigma, hidden = self.forward(decoder_input, hidden)

            # Zapisanie wyjść
            mu_seq[:, t, :] = mu
            sigma_seq[:, t, :] = sigma

            # Przygotowanie następnego inputu
            # Używamy porównania tensorów zamiast .item() żeby uniknąć synchronizacji CPU-GPU
            use_teacher_forcing = target is not None and torch.rand(1, device=device) < teacher_forcing_ratio
            if use_teacher_forcing:
                # Teacher forcing - używamy prawdziwej wartości
                decoder_input = target[:, t, :].unsqueeze(1)
            else:
                # Autoregresja - używamy predykcji (średniej)
                decoder_input = mu.unsqueeze(1)

        return mu_seq, sigma_seq

    def sample(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Próbkuje z rozkładu Gaussowskiego.

        Używa triku reparametryzacji dla zachowania gradientów.

        Args:
            mu: Średnie (batch_size, ..., output_size)
            sigma: Odchylenia standardowe (batch_size, ..., output_size)

        Returns:
            Próbki z rozkładu N(μ, σ²)
        """
        eps = torch.randn_like(mu)
        return mu + sigma * eps


if __name__ == "__main__":
    # Test modułu decoder
    print("Test GaussianDecoder")
    print("=" * 60)

    # Parametry
    batch_size = 32
    hidden_size = 64
    output_size = 2  # [x, v]
    num_layers = 2
    seq_len = 30

    # Tworzenie decodera
    decoder = GaussianDecoder(
        input_size=output_size,  # Autoregresja
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=0.1,
        min_sigma=1e-4
    )

    print(f"\nArchitektura decodera:")
    print(decoder)

    # Liczba parametrów
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nLiczba parametrów: {num_params:,}")

    # Przygotowanie danych testowych
    # Symulacja stanu ukrytego z encodera
    h_0 = torch.randn(num_layers, batch_size, hidden_size)
    c_0 = torch.randn(num_layers, batch_size, hidden_size)
    hidden = (h_0, c_0)

    # Początkowy input (ostatni krok z okna wejściowego)
    initial_input = torch.randn(batch_size, output_size)

    # Target dla teacher forcing
    target = torch.randn(batch_size, seq_len, output_size)

    # Test pojedynczego kroku
    print("\n" + "-" * 40)
    print("Test pojedynczego kroku:")

    x = initial_input.unsqueeze(1)  # (batch, 1, features)
    mu, sigma, new_hidden = decoder(x, hidden)

    print(f"Input shape: {x.shape}")
    print(f"mu shape: {mu.shape}")
    print(f"sigma shape: {sigma.shape}")
    print(f"sigma min: {sigma.min().item():.6f}")
    print(f"sigma max: {sigma.max().item():.6f}")

    # Sprawdzenie że sigma > 0
    assert (sigma > 0).all(), "Sigma musi być zawsze dodatnia!"
    print("✓ Sigma zawsze > 0")

    # Test generacji sekwencji (bez teacher forcing)
    print("\n" + "-" * 40)
    print("Test generacji sekwencji (autoregresja):")

    mu_seq, sigma_seq = decoder.forward_sequence(
        initial_input=initial_input,
        hidden=hidden,
        seq_len=seq_len,
        target=None,
        teacher_forcing_ratio=0.0
    )

    print(f"mu_seq shape: {mu_seq.shape}")
    print(f"sigma_seq shape: {sigma_seq.shape}")

    # Test z teacher forcing
    print("\n" + "-" * 40)
    print("Test z teacher forcing (ratio=0.5):")

    mu_seq_tf, sigma_seq_tf = decoder.forward_sequence(
        initial_input=initial_input,
        hidden=hidden,
        seq_len=seq_len,
        target=target,
        teacher_forcing_ratio=0.5
    )

    print(f"mu_seq_tf shape: {mu_seq_tf.shape}")
    print(f"sigma_seq_tf shape: {sigma_seq_tf.shape}")

    # Test próbkowania
    print("\n" + "-" * 40)
    print("Test próbkowania:")

    samples = decoder.sample(mu_seq, sigma_seq)
    print(f"Samples shape: {samples.shape}")

    # Sprawdzenie gradientów
    print("\n" + "-" * 40)
    print("Test gradientów:")

    loss = mu_seq.sum() + sigma_seq.sum()
    loss.backward()

    grad_exists = all(p.grad is not None for p in decoder.parameters())
    print(f"Gradienty obliczone: {grad_exists}")

    print("\n" + "=" * 60)
    print("Test zakończony pomyślnie!")
