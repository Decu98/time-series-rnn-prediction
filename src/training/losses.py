"""
Moduł funkcji straty.

Implementuje funkcje straty dla modelu probabilistycznego,
głównie Gaussian Negative Log-Likelihood (NLL).
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean',
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Oblicza Gaussian Negative Log-Likelihood.

    Dla rozkładu normalnego N(μ, σ²) log-likelihood wynosi:
        log p(x|μ,σ) = -0.5 * (log(2π) + 2*log(σ) + ((x-μ)/σ)²)

    NLL to -log p(x|μ,σ).

    Args:
        mu: Predykowane średnie, kształt (batch, ...)
        sigma: Predykowane odchylenia standardowe, kształt (batch, ...)
        target: Wartości docelowe, kształt (batch, ...)
        reduction: Typ redukcji ('none', 'mean', 'sum')
        eps: Małą wartość dla stabilności numerycznej

    Returns:
        Strata NLL (skalar jeśli reduction != 'none')
    """
    # Zabezpieczenie przed sigma = 0
    sigma = sigma.clamp(min=eps)

    # Log-likelihood
    # -log p(x|μ,σ) = 0.5 * (log(2π) + 2*log(σ) + ((x-μ)/σ)²)
    log_2pi = math.log(2 * math.pi)
    squared_error = ((target - mu) / sigma) ** 2
    log_sigma = torch.log(sigma)

    nll = 0.5 * (log_2pi + 2 * log_sigma + squared_error)

    # Redukcja
    if reduction == 'none':
        return nll
    elif reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    else:
        raise ValueError(f"Nieznany typ redukcji: {reduction}")


def multistep_gaussian_nll_loss(
    mu_seq: torch.Tensor,
    sigma_seq: torch.Tensor,
    target_seq: torch.Tensor,
    time_weights: Optional[torch.Tensor] = None,
    feature_weights: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Oblicza multi-step Gaussian NLL dla sekwencji predykcji.

    Agreguje stratę po całym horyzoncie predykcji T_out,
    z opcjonalnymi wagami czasowymi i cechowymi.

    Args:
        mu_seq: Predykowane średnie (batch, T_out, features)
        sigma_seq: Predykowane odchylenia std (batch, T_out, features)
        target_seq: Wartości docelowe (batch, T_out, features)
        time_weights: Opcjonalne wagi dla kroków czasowych (T_out,)
        feature_weights: Opcjonalne wagi dla cech (features,)
        reduction: Typ redukcji ('mean', 'sum')

    Returns:
        Skalarowa wartość straty
    """
    batch_size, T_out, features = mu_seq.shape

    # Obliczenie NLL dla każdego elementu
    nll = gaussian_nll_loss(
        mu=mu_seq,
        sigma=sigma_seq,
        target=target_seq,
        reduction='none'
    )  # (batch, T_out, features)

    # Aplikacja wag cechowych
    if feature_weights is not None:
        # feature_weights: (features,) -> (1, 1, features)
        feature_weights = feature_weights.view(1, 1, -1)
        nll = nll * feature_weights

    # Aplikacja wag czasowych
    if time_weights is not None:
        # time_weights: (T_out,) -> (1, T_out, 1)
        time_weights = time_weights.view(1, -1, 1)
        nll = nll * time_weights

    # Redukcja
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    else:
        raise ValueError(f"Nieznany typ redukcji: {reduction}")


def exponential_time_weights(
    T_out: int,
    decay: float = 0.01,
    normalize: bool = True,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generuje wykładniczo rosnące wagi czasowe.

    Późniejsze kroki predykcji mogą mieć większą wagę,
    ponieważ są trudniejsze do przewidzenia.

    Args:
        T_out: Długość sekwencji
        decay: Współczynnik wykładniczy
        normalize: Czy normalizować wagi do sumy = T_out
        device: Urządzenie dla tensora

    Returns:
        Tensor wag (T_out,)
    """
    t = torch.arange(T_out, dtype=torch.float32, device=device)
    weights = torch.exp(decay * t)

    if normalize:
        weights = weights * (T_out / weights.sum())

    return weights


def inverse_sigma_weights(
    sigma_seq: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Oblicza wagi proporcjonalne do 1/σ.

    Mniejsza niepewność = większa waga.

    Args:
        sigma_seq: Predykowane odchylenia std (batch, T_out, features)
        eps: Stabilność numeryczna

    Returns:
        Tensor wag (batch, T_out, features)
    """
    return 1.0 / (sigma_seq + eps)


class GaussianNLLLoss(nn.Module):
    """
    Moduł PyTorch dla Gaussian NLL loss.

    Opakowuje funkcję gaussian_nll_loss jako nn.Module
    dla wygodniejszej integracji z pipeline'em treningowym.

    Attributes:
        reduction: Typ redukcji
        eps: Stabilność numeryczna
    """

    def __init__(
        self,
        reduction: str = 'mean',
        eps: float = 1e-6
    ):
        """
        Inicjalizacja modułu straty.

        Args:
            reduction: Typ redukcji ('none', 'mean', 'sum')
            eps: Mała wartość dla stabilności
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Oblicza stratę.

        Args:
            mu: Predykowane średnie
            sigma: Predykowane odchylenia standardowe
            target: Wartości docelowe

        Returns:
            Wartość straty
        """
        return gaussian_nll_loss(
            mu=mu,
            sigma=sigma,
            target=target,
            reduction=self.reduction,
            eps=self.eps
        )


class MultiStepGaussianNLLLoss(nn.Module):
    """
    Moduł PyTorch dla multi-step Gaussian NLL loss.

    Attributes:
        time_weights: Wagi czasowe (opcjonalne)
        feature_weights: Wagi cechowe (opcjonalne)
        reduction: Typ redukcji
    """

    def __init__(
        self,
        T_out: Optional[int] = None,
        num_features: Optional[int] = None,
        use_time_weights: bool = False,
        time_decay: float = 0.01,
        feature_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Inicjalizacja modułu straty.

        Args:
            T_out: Długość sekwencji (wymagane jeśli use_time_weights=True)
            num_features: Liczba cech (opcjonalne)
            use_time_weights: Czy używać wykładniczych wag czasowych
            time_decay: Współczynnik wykładniczy dla wag czasowych
            feature_weights: Wagi dla poszczególnych cech
            reduction: Typ redukcji
        """
        super().__init__()

        self.reduction = reduction

        # Wagi czasowe
        if use_time_weights and T_out is not None:
            time_weights = exponential_time_weights(T_out, decay=time_decay)
            self.register_buffer('time_weights', time_weights)
        else:
            self.time_weights = None

        # Wagi cechowe
        if feature_weights is not None:
            self.register_buffer('feature_weights', feature_weights)
        else:
            self.feature_weights = None

    def forward(
        self,
        mu_seq: torch.Tensor,
        sigma_seq: torch.Tensor,
        target_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Oblicza stratę.

        Args:
            mu_seq: Predykowane średnie (batch, T_out, features)
            sigma_seq: Predykowane odchylenia std (batch, T_out, features)
            target_seq: Wartości docelowe (batch, T_out, features)

        Returns:
            Skalarowa wartość straty
        """
        return multistep_gaussian_nll_loss(
            mu_seq=mu_seq,
            sigma_seq=sigma_seq,
            target_seq=target_seq,
            time_weights=self.time_weights,
            feature_weights=self.feature_weights,
            reduction=self.reduction
        )


if __name__ == "__main__":
    # Test modułu losses
    print("Test funkcji straty Gaussian NLL")
    print("=" * 60)

    # Parametry testowe
    batch_size = 32
    T_out = 30
    features = 2

    # Symulacja predykcji i targetów
    mu = torch.randn(batch_size, T_out, features)
    sigma = torch.rand(batch_size, T_out, features) + 0.1  # sigma > 0
    target = torch.randn(batch_size, T_out, features)

    # Test podstawowej funkcji
    print("\n1. Test gaussian_nll_loss:")
    loss_none = gaussian_nll_loss(mu, sigma, target, reduction='none')
    loss_mean = gaussian_nll_loss(mu, sigma, target, reduction='mean')
    loss_sum = gaussian_nll_loss(mu, sigma, target, reduction='sum')

    print(f"   Kształt (reduction='none'): {loss_none.shape}")
    print(f"   Wartość (reduction='mean'): {loss_mean.item():.4f}")
    print(f"   Wartość (reduction='sum'): {loss_sum.item():.4f}")

    # Weryfikacja: dla idealnej predykcji (mu=target) strata powinna być mała
    print("\n2. Test z idealną predykcją (mu=target):")
    loss_perfect = gaussian_nll_loss(target, sigma, target, reduction='mean')
    print(f"   Strata (mu=target): {loss_perfect.item():.4f}")
    print(f"   Strata (mu≠target): {loss_mean.item():.4f}")
    assert loss_perfect < loss_mean, "Idealna predykcja powinna mieć mniejszą stratę!"
    print("   ✓ Idealna predykcja ma mniejszą stratę")

    # Test multi-step loss
    print("\n3. Test multistep_gaussian_nll_loss:")
    loss_multistep = multistep_gaussian_nll_loss(
        mu_seq=mu,
        sigma_seq=sigma,
        target_seq=target,
        reduction='mean'
    )
    print(f"   Wartość: {loss_multistep.item():.4f}")

    # Test z wagami czasowymi
    print("\n4. Test z wagami czasowymi:")
    time_weights = exponential_time_weights(T_out, decay=0.05)
    print(f"   Wagi czasowe: min={time_weights.min().item():.3f}, "
          f"max={time_weights.max().item():.3f}")

    loss_weighted = multistep_gaussian_nll_loss(
        mu_seq=mu,
        sigma_seq=sigma,
        target_seq=target,
        time_weights=time_weights,
        reduction='mean'
    )
    print(f"   Strata z wagami: {loss_weighted.item():.4f}")

    # Test modułu nn.Module
    print("\n5. Test GaussianNLLLoss (nn.Module):")
    criterion = GaussianNLLLoss(reduction='mean')
    loss_module = criterion(mu, sigma, target)
    print(f"   Wartość: {loss_module.item():.4f}")

    # Test MultiStepGaussianNLLLoss
    print("\n6. Test MultiStepGaussianNLLLoss (nn.Module):")
    criterion_multistep = MultiStepGaussianNLLLoss(
        T_out=T_out,
        use_time_weights=True,
        time_decay=0.02,
        reduction='mean'
    )
    loss_multistep_module = criterion_multistep(mu, sigma, target)
    print(f"   Wartość: {loss_multistep_module.item():.4f}")

    # Test gradientów
    print("\n7. Test gradientów:")
    mu_grad = mu.clone().requires_grad_(True)
    sigma_grad = sigma.clone().requires_grad_(True)

    loss = gaussian_nll_loss(mu_grad, sigma_grad, target, reduction='mean')
    loss.backward()

    print(f"   mu gradient shape: {mu_grad.grad.shape}")
    print(f"   sigma gradient shape: {sigma_grad.grad.shape}")
    print("   ✓ Gradienty obliczone poprawnie")

    print("\n" + "=" * 60)
    print("Test zakończony pomyślnie!")
