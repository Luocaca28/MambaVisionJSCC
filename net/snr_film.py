import torch
import torch.nn as nn


class SNRFiLM(nn.Module):
    """
    Lightweight SNR-conditioned FiLM module for JSCC bottleneck tokens.

    Input:
        x:   [B, L, C]
        snr: int / float / scalar tensor / [B] tensor

    Output:
        y:   [B, L, C]
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        snr_min: float = 0.0,
        snr_max: float = 20.0,
        film_scale: float = 0.1,
        identity_init: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.film_scale = float(film_scale)

        if self.snr_max <= self.snr_min:
            self.snr_max = self.snr_min + 1.0

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * self.dim),
        )

        # 关键：初始化成“近似恒等映射”
        # 一开始 gamma=0, beta=0，所以 y ≈ x
        # 这样不会刚开始训练就破坏原来的 JSCC 链路。
        if identity_init:
            last = self.mlp[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _make_snr_tensor(self, snr, batch_size: int, device, dtype):
        if isinstance(snr, torch.Tensor):
            snr_tensor = snr.to(device=device, dtype=dtype)
            if snr_tensor.dim() == 0:
                snr_tensor = snr_tensor.view(1).expand(batch_size)
            elif snr_tensor.numel() == 1:
                snr_tensor = snr_tensor.reshape(1).expand(batch_size)
            else:
                snr_tensor = snr_tensor.reshape(batch_size)
        else:
            snr_tensor = torch.full(
                (batch_size,),
                float(snr),
                device=device,
                dtype=dtype,
            )

        return snr_tensor.view(batch_size, 1)

    def forward(self, x: torch.Tensor, snr) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"SNRFiLM expects x with shape [B, L, C], got {x.shape}")

        B, L, C = x.shape
        if C != self.dim:
            raise ValueError(f"SNRFiLM dim mismatch: expected C={self.dim}, got C={C}")

        snr_tensor = self._make_snr_tensor(
            snr=snr,
            batch_size=B,
            device=x.device,
            dtype=x.dtype,
        )

        # 归一化到大约 [-1, 1]
        snr_norm = (snr_tensor - self.snr_min) / (self.snr_max - self.snr_min)
        snr_norm = snr_norm * 2.0 - 1.0

        gamma_beta = self.mlp(snr_norm)          # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(1)               # [B, 1, C]
        beta = beta.unsqueeze(1)                 # [B, 1, C]

        # 用 1 + scale * gamma，保证初始时接近恒等映射
        return x * (1.0 + self.film_scale * gamma) + self.film_scale * beta


class SNRLatentRefiner(nn.Module):
    """
    SNR-conditioned residual refiner for noisy received JSCC latent.

    x:   [B, L, C]
    snr: scalar / int / float / tensor
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        snr_min: float = 0.0,
        snr_max: float = 20.0,
        scale: float = 0.1,
    ):
        super().__init__()
        self.dim = int(dim)
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.scale = float(scale)

        if self.snr_max <= self.snr_min:
            self.snr_max = self.snr_min + 1.0

        self.snr_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.refine = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        # 初始为近似恒等映射，避免一开始破坏 baseline
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def _make_snr_tensor(self, snr, batch_size: int, device, dtype):
        if isinstance(snr, torch.Tensor):
            snr_tensor = snr.to(device=device, dtype=dtype)
            if snr_tensor.dim() == 0:
                snr_tensor = snr_tensor.view(1).expand(batch_size)
            elif snr_tensor.numel() == 1:
                snr_tensor = snr_tensor.reshape(1).expand(batch_size)
            else:
                snr_tensor = snr_tensor.reshape(batch_size)
        else:
            snr_tensor = torch.full(
                (batch_size,),
                float(snr),
                device=device,
                dtype=dtype,
            )
        return snr_tensor.view(batch_size, 1)

    def forward(self, x: torch.Tensor, snr) -> torch.Tensor:
        B, L, C = x.shape

        snr_tensor = self._make_snr_tensor(
            snr=snr,
            batch_size=B,
            device=x.device,
            dtype=x.dtype,
        )

        snr_norm = (snr_tensor - self.snr_min) / (self.snr_max - self.snr_min)
        snr_norm = snr_norm * 2.0 - 1.0

        snr_bias = self.snr_mlp(snr_norm).unsqueeze(1)  # [B, 1, C]

        # noisy latent correction
        residual = self.refine(x + snr_bias)

        return x + self.scale * residual
