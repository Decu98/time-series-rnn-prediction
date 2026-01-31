"""
Moduł encodera LSTM.

Encoder przetwarza sekwencję wejściową (historię) i generuje
reprezentację kontekstową w postaci stanów ukrytych LSTM.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    Encoder oparty na wielowarstwowym LSTM.

    Przetwarza sekwencję wejściową i zwraca końcowe stany
    (hidden state i cell state) reprezentujące zakodowany kontekst.

    Attributes:
        input_size: Liczba cech wejściowych
        hidden_size: Rozmiar warstwy ukrytej
        num_layers: Liczba warstw LSTM
        dropout: Współczynnik dropout między warstwami
        bidirectional: Czy LSTM ma być dwukierunkowy
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """
        Inicjalizacja encodera.

        Args:
            input_size: Liczba cech wejściowych (np. 2 dla [x, v])
            hidden_size: Rozmiar warstwy ukrytej LSTM
            num_layers: Liczba warstw LSTM
            dropout: Współczynnik dropout (stosowany między warstwami)
            bidirectional: Czy używać dwukierunkowego LSTM
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Warstwa LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Layer Normalization dla wyjściowych stanów
        # Stosujemy LN do stanów ukrytych
        self.layer_norm_h = nn.LayerNorm(hidden_size * self.num_directions)
        self.layer_norm_c = nn.LayerNorm(hidden_size * self.num_directions)

        # Opcjonalna projekcja dla dwukierunkowego LSTM
        # (redukuje wymiar z 2*hidden_size do hidden_size)
        if bidirectional:
            self.projection_h = nn.Linear(
                hidden_size * 2,
                hidden_size
            )
            self.projection_c = nn.Linear(
                hidden_size * 2,
                hidden_size
            )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Przetwarzanie sekwencji wejściowej przez encoder.

        Args:
            x: Tensor wejściowy (batch_size, seq_len, input_size)
            hidden: Opcjonalny początkowy stan ukryty (h_0, c_0)

        Returns:
            Tuple zawierający:
                - outputs: Wszystkie wyjścia LSTM (batch_size, seq_len, hidden_size * num_directions)
                - (h_n, c_n): Końcowe stany ukryte (dla decodera)
        """
        batch_size = x.size(0)

        # Inicjalizacja stanów ukrytych jeśli nie podano
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)

        # Przepuszczenie przez LSTM
        outputs, (h_n, c_n) = self.lstm(x, hidden)

        # Dla dwukierunkowego LSTM: łączenie kierunków
        if self.bidirectional:
            # h_n ma kształt (num_layers * num_directions, batch, hidden_size)
            # Przekształcamy do (num_layers, batch, hidden_size * num_directions)
            h_n = self._merge_bidirectional(h_n)
            c_n = self._merge_bidirectional(c_n)

            # Projekcja do oryginalnego hidden_size
            h_n = self.projection_h(h_n)
            c_n = self.projection_c(c_n)

        # Layer Normalization na końcowych stanach
        # Stosujemy do każdej warstwy osobno
        h_n_normalized = self._apply_layer_norm(h_n, self.layer_norm_h)
        c_n_normalized = self._apply_layer_norm(c_n, self.layer_norm_c)

        return outputs, (h_n_normalized, c_n_normalized)

    def _init_hidden(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inicjalizuje stany ukryte zerami.

        Args:
            batch_size: Rozmiar batcha
            device: Urządzenie (CPU/GPU)

        Returns:
            Tuple (h_0, c_0) z zerowymi tensorami
        """
        num_directions = self.num_directions

        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )

        c_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )

        return (h_0, c_0)

    def _merge_bidirectional(self, states: torch.Tensor) -> torch.Tensor:
        """
        Łączy stany z obu kierunków dwukierunkowego LSTM.

        Args:
            states: Tensor (num_layers * 2, batch, hidden_size)

        Returns:
            Tensor (num_layers, batch, hidden_size * 2)
        """
        # Przekształcenie: (num_layers * 2, batch, hidden) -> (num_layers, 2, batch, hidden)
        states = states.view(self.num_layers, 2, states.size(1), self.hidden_size)

        # Konkatenacja kierunków: (num_layers, batch, hidden * 2)
        states = torch.cat([states[:, 0, :, :], states[:, 1, :, :]], dim=2)

        return states

    def _apply_layer_norm(
        self,
        states: torch.Tensor,
        layer_norm: nn.LayerNorm
    ) -> torch.Tensor:
        """
        Stosuje Layer Normalization do stanów LSTM.

        Args:
            states: Tensor (num_layers, batch, hidden_size)
            layer_norm: Moduł LayerNorm

        Returns:
            Znormalizowany tensor
        """
        # states ma kształt (num_layers, batch, hidden_size)
        # LN wymaga ostatniego wymiaru zgodnego z normalized_shape
        normalized = layer_norm(states)
        return normalized

    def get_output_size(self) -> int:
        """
        Zwraca rozmiar wyjściowy encodera (dla decodera).

        Returns:
            Rozmiar hidden state (po projekcji dla bidirectional)
        """
        return self.hidden_size


if __name__ == "__main__":
    # Test modułu encoder
    print("Test LSTMEncoder")
    print("=" * 60)

    # Parametry
    batch_size = 32
    seq_len = 50
    input_size = 2
    hidden_size = 64
    num_layers = 2

    # Tworzenie encodera
    encoder = LSTMEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.1,
        bidirectional=False
    )

    print(f"\nArchitektura encodera:")
    print(encoder)

    # Liczba parametrów
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nLiczba parametrów: {num_params:,}")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_size)
    print(f"\nInput shape: {x.shape}")

    outputs, (h_n, c_n) = encoder(x)

    print(f"Outputs shape: {outputs.shape}")
    print(f"h_n shape: {h_n.shape}")
    print(f"c_n shape: {c_n.shape}")

    # Test z bidirectional
    print("\n" + "-" * 40)
    print("Test z bidirectional=True:")

    encoder_bi = LSTMEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.1,
        bidirectional=True
    )

    outputs_bi, (h_n_bi, c_n_bi) = encoder_bi(x)

    print(f"Outputs shape: {outputs_bi.shape}")
    print(f"h_n shape: {h_n_bi.shape}")
    print(f"c_n shape: {c_n_bi.shape}")

    # Sprawdzenie gradientów
    print("\n" + "-" * 40)
    print("Test gradientów:")

    loss = outputs.sum() + h_n.sum()
    loss.backward()

    grad_exists = all(p.grad is not None for p in encoder.parameters())
    print(f"Gradienty obliczone: {grad_exists}")

    print("\n" + "=" * 60)
    print("Test zakończony pomyślnie!")
