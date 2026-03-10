# MIFNO/train_MIFNO.py
# ==========================================
# TRUE Multi-scale Input Fourier Neural Operator (MIFNO)
#
# Architecture:
#   - 4 scalar parameters → 4 independent input "functions" via learned lifting
#   - Each input function is encoded at MULTIPLE SCALES (coarse → fine)
#   - Multi-scale features are fused and processed by stacked FNO blocks
#   - Dual output head predicts temperature + stress fields over (Nx, Nt)
#
# Key differences from plain FNO:
#   1. Each of the 4 inputs is treated as a separate input channel/function
#   2. Multiscale encoder: each input is projected at S different resolutions
#   3. Cross-scale attention fusion aggregates multiscale representations
#   4. Input functions are injected at EACH FNO layer (not just the first)
#
# Supports: CPU / Apple MPS / CUDA
# Usage:
#   python MIFNO/train_MIFNO.py --device auto
#   python MIFNO/train_MIFNO.py --device mps
#   python MIFNO/train_MIFNO.py --device cuda
#   python MIFNO/train_MIFNO.py --device cpu --threads 8
# ==========================================

import os, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.fft import rfft2, irfft2
from torch.utils.data import TensorDataset, DataLoader


# ─────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────
def pick_device(req: str) -> torch.device:
    req = req.lower().strip()
    if req == "cpu":
        return torch.device("cpu")
    if req in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")
    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        raise RuntimeError(f"MPS requested but not available. mps_built={built}")
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError("device must be: auto | cpu | cuda | mps")


parser = argparse.ArgumentParser()
parser.add_argument("--device",  type=str, default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
parser.add_argument("--threads", type=int, default=8,
                    help="CPU threads (only used when device=cpu)")
args = parser.parse_args()

device = pick_device(args.device)
if device.type == "cpu":
    torch.set_num_threads(args.threads)

torch.manual_seed(42)
np.random.seed(42)

print(f"🚀 Training | device={device}")
print(f"   torch={torch.__version__} | cuda={torch.cuda.is_available()} | "
      f"mps={hasattr(torch.backends,'mps') and torch.backends.mps.is_available()}")
if device.type == "cuda":
    print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
elif device.type == "mps":
    print("   Using Apple MPS backend")


# ─────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────
Nx, Nt        = 49, 500
param_dim     = 4           # number of scalar input parameters
n_scales      = 3           # multiscale levels per input (MIFNO key feature)
scale_width   = 16          # channels per scale per input
width         = 64          # main FNO channel width  (= param_dim * n_scales * scale_width / something)
depth         = 6           # number of FNO blocks
modes_x       = 16
modes_t       = 16
lr            = 1e-3
weight_decay  = 1e-5
epochs        = 500
batch_size    = 8

results_dir   = "results/train"
save_dir      = "MIFNO"
save_path     = os.path.join(save_dir, f"mifno_model_dual_{device.type}.pt")
best_path     = os.path.join(save_dir, f"mifno_model_dual_{device.type}_best.pt")
os.makedirs(save_dir, exist_ok=True)

cfg = dict(Nx=Nx, Nt=Nt, param_dim=param_dim, n_scales=n_scales,
           scale_width=scale_width, width=width, depth=depth,
           modes_x=modes_x, modes_t=modes_t)
with open(os.path.join(save_dir, "mifno_config.json"), "w") as f:
    json.dump(cfg, f, indent=4)


# ─────────────────────────────────────────
# 1) Load & shape data
# ─────────────────────────────────────────
param_files  = sorted([f for f in os.listdir(results_dir) if f.startswith("params_case_")])
temp_files   = sorted([f for f in os.listdir(results_dir) if f.startswith("temperature_all_case")])
stress_files = sorted([f for f in os.listdir(results_dir) if f.startswith("stress_all_case")])

if not (len(param_files) == len(temp_files) == len(stress_files) > 0):
    raise RuntimeError(
        f"Mismatch/empty dataset in {results_dir}: "
        f"params={len(param_files)}, temps={len(temp_files)}, stress={len(stress_files)}"
    )

X_params, Y_temperatures, Y_stresses = [], [], []
for p, t, s in zip(param_files, temp_files, stress_files):
    params   = np.loadtxt(os.path.join(results_dir, p), skiprows=1, delimiter=",")
    temps    = np.loadtxt(os.path.join(results_dir, t))
    stresses = np.loadtxt(os.path.join(results_dir, s))
    if params.ndim == 0:
        params = np.expand_dims(params, axis=0)
    X_params.append(params)
    Y_temperatures.append(temps)
    Y_stresses.append(stresses)

N        = len(X_params)
X        = np.stack(X_params).astype(np.float32)                                      # (N, 4)
Y_temp   = np.stack(Y_temperatures).astype(np.float32).reshape(N, Nt, Nx).transpose(0, 2, 1)  # (N, Nx, Nt)
Y_strs   = np.stack(Y_stresses).astype(np.float32).reshape(N, Nt, Nx).transpose(0, 2, 1)      # (N, Nx, Nt)


# ─────────────────────────────────────────
# 2) Normalize & save scalers
# ─────────────────────────────────────────
x_scaler  = StandardScaler()
X_scaled  = x_scaler.fit_transform(X)

yT_scaler = StandardScaler()
Y_temp_s  = yT_scaler.fit_transform(Y_temp.reshape(N, -1)).reshape(N, Nx, Nt)

yS_scaler = StandardScaler()
Y_strs_s  = yS_scaler.fit_transform(Y_strs.reshape(N, -1)).reshape(N, Nx, Nt)

Y_combined = np.stack([Y_temp_s, Y_strs_s], axis=1)  # (N, 2, Nx, Nt)

np.save(os.path.join(save_dir, "X_train_mifno.npy"),           X_scaled)
np.save(os.path.join(save_dir, "Y_train_mifno_combined.npy"),  Y_combined)
np.save(os.path.join(save_dir, "x_scaler_mean_mifno.npy"),     x_scaler.mean_)
np.save(os.path.join(save_dir, "x_scaler_scale_mifno.npy"),    x_scaler.scale_)
np.save(os.path.join(save_dir, "y_temp_scaler_mean.npy"),      yT_scaler.mean_.reshape(Nx, Nt))
np.save(os.path.join(save_dir, "y_temp_scaler_scale.npy"),     yT_scaler.scale_.reshape(Nx, Nt))
np.save(os.path.join(save_dir, "y_stress_scaler_mean.npy"),    yS_scaler.mean_.reshape(Nx, Nt))
np.save(os.path.join(save_dir, "y_stress_scaler_scale.npy"),   yS_scaler.scale_.reshape(Nx, Nt))


# ═══════════════════════════════════════════════════════════════
#  TRUE MIFNO ARCHITECTURE
#
#  Core idea: each of the 4 scalar inputs is treated as an
#  independent "input function" and lifted to the (Nx, Nt) domain
#  at MULTIPLE spatial/temporal scales.  The multi-scale features
#  are fused before entering stacked FNO blocks, and a learnable
#  conditioning signal from each input is injected at every layer.
# ═══════════════════════════════════════════════════════════════

class MultiscaleInputEncoder(nn.Module):
    """
    Encodes ONE scalar input parameter into n_scales field representations
    over the (Nx, Nt) domain at different effective resolutions.

    Scale k uses resolution (Nx // 2^k, Nt // 2^k) and upsamples back,
    so coarser scales capture global structure while finer scales
    capture local detail — the defining feature of MIFNO.
    """
    def __init__(self, Nx: int, Nt: int, n_scales: int, scale_width: int):
        super().__init__()
        self.Nx, self.Nt     = Nx, Nt
        self.n_scales        = n_scales
        self.scale_width     = scale_width

        # Each scale: MLP scalar → small field at that scale's resolution
        self.scale_lifters = nn.ModuleList()
        self.scale_convs   = nn.ModuleList()
        for k in range(n_scales):
            nx_k = max(Nx // (2 ** k), 4)
            nt_k = max(Nt // (2 ** k), 4)
            self.scale_lifters.append(
                nn.Sequential(
                    nn.Linear(1, 64), nn.GELU(),
                    nn.Linear(64, nx_k * nt_k),
                )
            )
            self.scale_convs.append(
                nn.Sequential(
                    nn.Conv2d(1, scale_width, kernel_size=3, padding=1), nn.GELU(),
                    nn.Conv2d(scale_width, scale_width, kernel_size=3, padding=1),
                )
            )

    def forward(self, x_scalar: torch.Tensor) -> torch.Tensor:
        """
        x_scalar: (B, 1)  — one normalised parameter value
        returns:  (B, n_scales * scale_width, Nx, Nt)
        """
        B = x_scalar.shape[0]
        scale_fields = []
        for k in range(self.n_scales):
            nx_k = max(self.Nx // (2 ** k), 4)
            nt_k = max(self.Nt // (2 ** k), 4)

            # lift scalar → coarse field
            field_k = self.scale_lifters[k](x_scalar)          # (B, nx_k*nt_k)
            field_k = field_k.view(B, 1, nx_k, nt_k)           # (B, 1, nx_k, nt_k)

            # convolve at coarse resolution
            field_k = self.scale_convs[k](field_k)              # (B, scale_width, nx_k, nt_k)

            # upsample back to full (Nx, Nt)
            field_k = F.interpolate(field_k, size=(self.Nx, self.Nt),
                                    mode="bilinear", align_corners=False)
            scale_fields.append(field_k)

        return torch.cat(scale_fields, dim=1)   # (B, n_scales*scale_width, Nx, Nt)


class CrossScaleFusion(nn.Module):
    """
    Fuses the multi-scale, multi-input feature maps into a single
    width-channel representation.

    Input:  (B, param_dim * n_scales * scale_width, Nx, Nt)
    Output: (B, width, Nx, Nt)

    Uses channel attention (squeeze-excitation style) so the network
    learns which input × scale combinations matter most.
    """
    def __init__(self, in_ch: int, width: int, Nx: int, Nt: int):
        super().__init__()
        self.conv1   = nn.Conv2d(in_ch, width, kernel_size=1)
        self.conv2   = nn.Conv2d(width,  width, kernel_size=3, padding=1)
        self.norm    = nn.GroupNorm(8, width)

        # squeeze-excitation on channels
        self.se_fc1  = nn.Linear(width, width // 4)
        self.se_fc2  = nn.Linear(width // 4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.norm(self.conv2(x)))

        # channel attention
        se = x.mean(dim=(2, 3))                   # (B, width) global avg pool
        se = F.gelu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        x  = x * se.unsqueeze(-1).unsqueeze(-1)   # channel-wise rescaling

        return x


class SpectralConv2d(nn.Module):
    """Standard FNO spectral convolution (unchanged from original)."""
    def __init__(self, in_ch: int, out_ch: int, modes_x: int = 16, modes_t: int = 16):
        super().__init__()
        self.modes_x, self.modes_t = modes_x, modes_t
        self.weight = nn.Parameter(
            torch.randn(in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat)
            / (in_ch * out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, nx, nt = x.shape
        x_ft  = rfft2(x, s=(nx, nt), norm="forward")
        out_ft = torch.zeros(B, self.weight.shape[1], nx, nt // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes_x, :self.modes_t] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes_x, :self.modes_t],
            self.weight,
        )
        return irfft2(out_ft, s=(nx, nt), norm="forward")


class MIFNOBlock(nn.Module):
    """
    FNO block with INPUT INJECTION — the fused multi-scale input
    representation is added at every layer so that physical parameter
    information is never lost deep in the network (key MIFNO principle).
    """
    def __init__(self, ch: int, cond_ch: int, modes_x: int = 16, modes_t: int = 16):
        super().__init__()
        self.spec     = SpectralConv2d(ch, ch, modes_x, modes_t)
        self.lin      = nn.Conv2d(ch, ch, 1)
        self.norm     = nn.GroupNorm(8, ch)
        # projects conditioning signal to match width
        self.cond_proj = nn.Conv2d(cond_ch, ch, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, ch,      Nx, Nt)  — current feature map
        cond: (B, cond_ch, Nx, Nt)  — fused multi-scale input signal
        """
        return F.gelu(
            self.norm(
                self.spec(x) + self.lin(x) + self.cond_proj(cond)
            )
        )


class MIFNO(nn.Module):
    """
    Multi-scale Input Fourier Neural Operator (MIFNO)

    Inputs
    ------
    x_vec : (B, 4)  — 4 normalised scalar parameters

    Outputs
    -------
    (B, 2, Nx, Nt)  — temperature field (ch 0) + stress field (ch 1)

    Architecture summary
    --------------------
    1. MultiscaleInputEncoder × 4  — each parameter → (B, n_scales*scale_width, Nx, Nt)
    2. CrossScaleFusion            — (B, 4*n_scales*scale_width, Nx, Nt) → (B, width, Nx, Nt)
    3. Stack of MIFNOBlock × depth — cond injected at EVERY layer
    4. Conv2d head                 — (B, width, Nx, Nt) → (B, 2, Nx, Nt)
    """
    def __init__(self,
                 Nx: int, Nt: int,
                 param_dim: int   = 4,
                 n_scales: int    = 3,
                 scale_width: int = 16,
                 width: int       = 64,
                 depth: int       = 6,
                 modes_x: int     = 16,
                 modes_t: int     = 16):
        super().__init__()
        self.param_dim = param_dim

        # ── 1. Per-input multiscale encoders ──────────────────────────────
        self.input_encoders = nn.ModuleList([
            MultiscaleInputEncoder(Nx, Nt, n_scales, scale_width)
            for _ in range(param_dim)
        ])

        # ── 2. Cross-scale fusion ─────────────────────────────────────────
        in_ch = param_dim * n_scales * scale_width
        self.fusion = CrossScaleFusion(in_ch, width, Nx, Nt)

        # ── 3. Stacked MIFNO blocks with conditioning ─────────────────────
        self.blocks = nn.ModuleList([
            MIFNOBlock(width, width, modes_x, modes_t)
            for _ in range(depth)
        ])

        # ── 4. Output projection ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 2, kernel_size=1),
        )

    def forward(self, x_vec: torch.Tensor) -> torch.Tensor:
        """x_vec: (B, param_dim)"""
        # encode each parameter independently at multiple scales
        per_input_fields = [
            enc(x_vec[:, i:i+1])          # (B, n_scales*scale_width, Nx, Nt)
            for i, enc in enumerate(self.input_encoders)
        ]
        # concatenate across inputs
        multi_input = torch.cat(per_input_fields, dim=1)  # (B, 4*n_s*sw, Nx, Nt)

        # fuse into a single width-channel conditioning signal
        cond = self.fusion(multi_input)    # (B, width, Nx, Nt)

        # process with FNO blocks — inject cond at every layer
        x = cond
        for blk in self.blocks:
            x = blk(x, cond)

        return self.head(x)                # (B, 2, Nx, Nt)


# ─────────────────────────────────────────
# 4) Training loop
# ─────────────────────────────────────────
model = MIFNO(
    Nx=Nx, Nt=Nt,
    param_dim=param_dim,
    n_scales=n_scales,
    scale_width=scale_width,
    width=width,
    depth=depth,
    modes_x=modes_x,
    modes_t=modes_t,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ MIFNO model | trainable params: {total_params:,}")

opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, factor=0.8, patience=80, threshold=1e-4
)

X_train = torch.tensor(X_scaled,   dtype=torch.float32)
Y_train = torch.tensor(Y_combined, dtype=torch.float32)

dataset = TensorDataset(X_train, Y_train)
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

best_loss = float("inf")

print(f"\nStarting training | epochs={epochs} | batch={batch_size}\n")
t0 = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    running, running_T, running_S = 0.0, 0.0, 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)

        pred = model(xb)                              # (B, 2, Nx, Nt)

        loss_T = F.mse_loss(pred[:, 0], yb[:, 0])

        loss_S = F.smooth_l1_loss(
            pred[:, 1].contiguous(),
            yb[:, 1].contiguous(),
            beta=0.5,
        )
        loss_S_energy = F.mse_loss(
            pred[:, 1].abs().mean(dim=(1, 2)),
            yb[:, 1].abs().mean(dim=(1, 2)),
        )

        loss = 0.4 * loss_T + 0.6 * loss_S + 0.1 * loss_S_energy
        loss.backward()
        opt.step()

        bs          = xb.size(0)
        running    += loss.item()   * bs
        running_T  += loss_T.item() * bs
        running_S  += loss_S.item() * bs

    epoch_loss = running   / len(dataset)
    epoch_T    = running_T / len(dataset)
    epoch_S    = running_S / len(dataset)
    sched.step(epoch_loss)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_path)

    if epoch % 50 == 0 or epoch == 1:
        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch:4d} | Loss={epoch_loss:.6e} "
              f"(T={epoch_T:.6e}, S={epoch_S:.6e}) | lr={lr_now:.2e}")

print(f"\n⏱️  Training time: {time.time() - t0:.2f}s")
torch.save(model.state_dict(), save_path)
print(f"✅ Saved LAST  weights → {save_path}")
print(f"🏁 Saved BEST  weights → {best_path}  (best_loss={best_loss:.6e})")
