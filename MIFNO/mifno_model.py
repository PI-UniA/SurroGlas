# MIFNO/mifno_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2, irfft2
from typing import Optional


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer over (x, t): (B, C, Nx, Nt) -> (B, C, Nx, Nt)
    Keeps only the lowest (modes_x, modes_t) frequency modes.
    """
    def __init__(self, in_ch: int, out_ch: int, modes_x: int = 16, modes_t: int = 16):
        super().__init__()
        self.modes_x = modes_x
        self.modes_t = modes_t
        self.weight = nn.Parameter(
            torch.randn(in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat) / (in_ch * out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Nx, Nt = x.shape
        x_ft = rfft2(x, s=(Nx, Nt), norm="forward")  # (B, C, Nx, Nt//2 + 1)
        out_ft = torch.zeros(B, self.weight.shape[1], Nx, Nt // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes_x, :self.modes_t] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes_x, :self.modes_t], self.weight
        )
        return irfft2(out_ft, s=(Nx, Nt), norm="forward")


class FNOBlock(nn.Module):
    """
    One FNO block: SpectralConv + 1x1 Conv residual + BatchNorm + GELU
    """
    def __init__(self, ch: int, modes_x: int = 16, modes_t: int = 16):
        super().__init__()
        self.spec = SpectralConv2d(ch, ch, modes_x, modes_t)
        self.lin  = nn.Conv2d(ch, ch, kernel_size=1)
        self.bn   = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.bn(self.spec(x) + self.lin(x)))


class SimpleMIFNO(nn.Module):
    """
    Vector → latent field via FC, then several FNO blocks, then 2-channel head.
    Outputs channels:
      - channel 0: Temperature T(x, t)
      - channel 1: Stress      σ(x, t)
    """
    def __init__(
        self,
        Nx: int,
        Nt: int,
        param_dim: int = 4,
        width: int = 32,
        depth: int = 6,
        modes_x: int = 16,
        modes_t: int = 16,
    ):
        super().__init__()
        self.Nx, self.Nt = Nx, Nt
        self.fc1 = nn.Linear(param_dim, 128)
        self.fc2 = nn.Linear(128, width * Nx * Nt)
        self.post = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1), nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes_x=modes_x, modes_t=modes_t) for _ in range(depth)]
        )
        # IMPORTANT: name stays "out" (backend can auto-map head↔out if needed)
        self.out = nn.Conv2d(width, 2, kernel_size=1)

    def forward(self, x_vec: torch.Tensor) -> torch.Tensor:
        """
        x_vec: (B, param_dim) — scaled process parameters
        returns: (B, 2, Nx, Nt) — channels: [T_hat, sigma_hat]
        """
        B = x_vec.shape[0]
        x = F.gelu(self.fc1(x_vec))
        x = self.fc2(x).view(B, -1, self.Nx, self.Nt)  # (B, width, Nx, Nt)
        x = self.post(x)
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)


# ---------- Convenience helpers for FastAPI backends ----------
def build_mifno(Nx: int, Nt: int,
                param_dim: int = 4,
                width: int = 32,
                depth: int = 6,
                modes_x: int = 16,
                modes_t: int = 16,
                device: Optional[torch.device] = None) -> SimpleMIFNO:
    device = device or torch.device("cpu")
    model = SimpleMIFNO(Nx, Nt, param_dim, width, depth, modes_x, modes_t)
    model.to(device).eval()
    return model


def load_weights_adapt_head(model: nn.Module, ckpt_path: str, map_location: str = "cpu") -> None:
    """
    Robust loader that adapts 'head' <-> 'out' naming differences if needed.
    """
    state = torch.load(ckpt_path, map_location=map_location)
    mkeys = set(model.state_dict().keys())
    ckeys = set(state.keys())
    has_head_ckpt = any(k.startswith("head.") for k in ckeys)
    has_out_ckpt  = any(k.startswith("out.")  for k in ckeys)
    has_head_mod  = any(k.startswith("head.") for k in mkeys)
    has_out_mod   = any(k.startswith("out.")  for k in mkeys)

    if has_head_ckpt and has_out_mod and not has_head_mod:
        state = {("out."+k[5:]) if k.startswith("head.") else k: v for k, v in state.items()}
    if has_out_ckpt and has_head_mod and not has_out_mod:
        state = {("head."+k[4:]) if k.startswith("out.") else k: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
