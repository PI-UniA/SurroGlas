import os
import re
import json
import time
import argparse
import datetime
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.fft import rfft2, irfft2


# ============================================================
# Device utilities
# ============================================================
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


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
parser.add_argument("--threads", type=int, default=8,
                    help="CPU threads, only used for device=cpu")
parser.add_argument("--results_dir", type=str, default="results/train")
parser.add_argument("--save_dir", type=str, default="MIFNO")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=16)          # was 4
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--val_ratio", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
# --- model ---
parser.add_argument("--width", type=int, default=64)               # was 96
parser.add_argument("--depth", type=int, default=6)                # was 8
parser.add_argument("--n_scales", type=int, default=3)
parser.add_argument("--scale_width", type=int, default=16)
parser.add_argument("--modes_x", type=int, default=12)
parser.add_argument("--modes_t", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.1)          # new: regularization
# --- FEM mesh (explicit, never guessed) ---
parser.add_argument("--Nx", type=int, default=29,
                    help="Number of spatial nodes from FEM mesh (through thickness)")
parser.add_argument("--Nt", type=int, default=3300,
                    help="Number of time steps from FEM output")
# --- training ---
parser.add_argument("--num_workers", type=int, default=0)          # 0 = main process only (safe without if __name__=='__main__')
parser.add_argument("--compile", action="store_true",              # new
                    help="Use torch.compile (PyTorch 2.x+)")
parser.add_argument("--save_every", type=int, default=25,          # new
                    help="Save history/config every N epochs")
args = parser.parse_args()

device = pick_device(args.device)
if device.type == "cpu":
    torch.set_num_threads(args.threads)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

print(f"🚀 Training MIFNO | device={device}")
print(f"   torch={torch.__version__} | cuda={torch.cuda.is_available()} | "
      f"mps={hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
if device.type == "cuda":
    print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
elif device.type == "mps":
    print("   Using Apple MPS backend")


# ============================================================
# File utilities
# ============================================================
def extract_case_idx(fname: str) -> int:
    m = re.search(r"case[_]?(\d+)|case(\d+)", fname)
    if m:
        for g in m.groups():
            if g is not None:
                return int(g)
    return 10**9


def load_param_file(path: str):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 0:
        data = np.expand_dims(data, axis=0)
    with open(path, "r") as f:
        first_line = f.readline().strip()
    header = [h.strip() for h in first_line.split(",") if h.strip()]
    return data.astype(np.float32), header


# ============================================================
# Load and shape dataset
# ============================================================
results_dir = args.results_dir
save_dir    = args.save_dir
Nx          = args.Nx
Nt          = args.Nt

os.makedirs(save_dir, exist_ok=True)

param_files = sorted(
    [f for f in os.listdir(results_dir) if f.startswith("params_case_")],
    key=extract_case_idx
)
temp_files = sorted(
    [f for f in os.listdir(results_dir) if f.startswith("temperature_all_case")],
    key=extract_case_idx
)
stress_files = sorted(
    [f for f in os.listdir(results_dir) if f.startswith("stress_all_case")],
    key=extract_case_idx
)

if not (len(param_files) == len(temp_files) == len(stress_files) > 0):
    raise RuntimeError(
        f"Mismatch/empty dataset in {results_dir}: "
        f"params={len(param_files)}, temps={len(temp_files)}, stress={len(stress_files)}"
    )

# Validate flat size against explicit FEM dimensions
first_temp = np.loadtxt(os.path.join(results_dir, temp_files[0]))
expected_flat = Nx * Nt
if first_temp.size != expected_flat:
    raise ValueError(
        f"FEM shape mismatch: --Nx={Nx} * --Nt={Nt} = {expected_flat}, "
        f"but first temperature file has {first_temp.size} elements. "
        f"Check your --Nx and --Nt arguments."
    )

X_params       = []
Y_temperatures = []
Y_stresses     = []
param_names    = None

for p, t, s in zip(param_files, temp_files, stress_files):
    params, header = load_param_file(os.path.join(results_dir, p))
    temps = np.loadtxt(os.path.join(results_dir, t)).astype(np.float32)
    strs  = np.loadtxt(os.path.join(results_dir, s)).astype(np.float32)

    if param_names is None:
        param_names = header
    elif header != param_names:
        raise ValueError(f"Parameter header mismatch in {p}")

    if temps.size != Nx * Nt:
        raise ValueError(f"Temperature size mismatch in {t}: got {temps.size}, expected {Nx*Nt}")
    if strs.size != Nx * Nt:
        raise ValueError(f"Stress size mismatch in {s}: got {strs.size}, expected {Nx*Nt}")

    X_params.append(params)
    Y_temperatures.append(temps.reshape(Nt, Nx).T)   # (Nx, Nt)
    Y_stresses.append(strs.reshape(Nt, Nx).T)        # (Nx, Nt)

X      = np.stack(X_params).astype(np.float32)
Y_temp = np.stack(Y_temperatures).astype(np.float32)
Y_strs = np.stack(Y_stresses).astype(np.float32)

N         = X.shape[0]
param_dim = X.shape[1]

print("\n✅ Dataset loaded")
print(f"   results_dir = {results_dir}")
print(f"   N           = {N}")
print(f"   param_dim   = {param_dim}")
print(f"   Nx, Nt      = {Nx}, {Nt}  (from --Nx/--Nt, validated against files)")
print(f"   param_names = {param_names}")


# ============================================================
# Unique run naming
# ============================================================
run_id    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = f"mifno_multizone_Nx{Nx}_Nt{Nt}_N{N}_{run_id}"

save_path_last  = os.path.join(save_dir, f"{base_name}_last.pt")
save_path_best  = os.path.join(save_dir, f"{base_name}_best.pt")
history_path    = os.path.join(save_dir, f"{base_name}_history.json")
cfg_path        = os.path.join(save_dir, f"{base_name}_config.json")

latest_last_path    = os.path.join(save_dir, "latest_model_last.pt")
latest_best_path    = os.path.join(save_dir, "latest_model_best.pt")
latest_cfg_path     = os.path.join(save_dir, "latest_config.json")
latest_history_path = os.path.join(save_dir, "latest_history.json")


# ============================================================
# Train/validation split
# ============================================================
idx_all = np.arange(N)
idx_train, idx_val = train_test_split(
    idx_all, test_size=args.val_ratio, random_state=args.seed, shuffle=True
)

X_train_raw = X[idx_train]
X_val_raw   = X[idx_val]

Y_temp_train_raw = Y_temp[idx_train]
Y_temp_val_raw   = Y_temp[idx_val]

Y_strs_train_raw = Y_strs[idx_train]
Y_strs_val_raw   = Y_strs[idx_val]


# ============================================================
# Normalization
# ============================================================
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train_raw)
X_val   = x_scaler.transform(X_val_raw)

# Global normalization per output field (not per pixel)
# Preserves spatial gradients unlike flattened StandardScaler
T_mean  = Y_temp_train_raw.mean()
T_std   = Y_temp_train_raw.std() + 1e-8
S_mean  = Y_strs_train_raw.mean()
S_std   = Y_strs_train_raw.std() + 1e-8

Y_temp_train = (Y_temp_train_raw - T_mean) / T_std
Y_temp_val   = (Y_temp_val_raw   - T_mean) / T_std
Y_strs_train = (Y_strs_train_raw - S_mean) / S_std
Y_strs_val   = (Y_strs_val_raw   - S_mean) / S_std

Y_train_combined = np.stack([Y_temp_train, Y_strs_train], axis=1)   # (N_tr, 2, Nx, Nt)
Y_val_combined   = np.stack([Y_temp_val,   Y_strs_val],   axis=1)   # (N_val, 2, Nx, Nt)

# Save scalers
np.save(os.path.join(save_dir, "x_scaler_mean_mifno_mz.npy"),  x_scaler.mean_)
np.save(os.path.join(save_dir, "x_scaler_scale_mifno_mz.npy"), x_scaler.scale_)
np.save(os.path.join(save_dir, "y_temp_mean_mz.npy"),  np.array([T_mean]))
np.save(os.path.join(save_dir, "y_temp_std_mz.npy"),   np.array([T_std]))
np.save(os.path.join(save_dir, "y_stress_mean_mz.npy"), np.array([S_mean]))
np.save(os.path.join(save_dir, "y_stress_std_mz.npy"),  np.array([S_std]))

np.save(os.path.join(save_dir, "X_train_mifno_mz.npy"),          X_train)
np.save(os.path.join(save_dir, "Y_train_mifno_combined_mz.npy"),  Y_train_combined)
np.save(os.path.join(save_dir, "X_val_mifno_mz.npy"),            X_val)
np.save(os.path.join(save_dir, "Y_val_mifno_combined_mz.npy"),    Y_val_combined)

print("✅ Saved scalers (global field normalization)")


# ============================================================
# Model
# ============================================================

class BatchedMultiscaleEncoder(nn.Module):
    """
    Memory-efficient multiscale encoder.

    Strategy:
      1. Small MLP: (B, param_dim) → (B, embed_dim)          [tiny, ~thousands of params]
      2. Per-scale head: (B, embed_dim) → (B, scale_width, BASE_NX, BASE_NT)
                         using a fixed small base grid (e.g. 8×16)
      3. Single F.interpolate per scale up to full (Nx, Nt)
      4. Lightweight Conv2d refinement at full resolution

    Total params: ~hundreds of thousands, not hundreds of millions.
    """
    BASE_NX = 8    # small base spatial grid — upsampled to Nx
    BASE_NT = 16   # small base time grid   — upsampled to Nt

    def __init__(self, Nx: int, Nt: int, param_dim: int,
                 n_scales: int, scale_width: int, embed_dim: int = 128):
        super().__init__()
        self.Nx          = Nx
        self.Nt          = Nt
        self.param_dim   = param_dim
        self.n_scales    = n_scales
        self.scale_width = scale_width
        self.embed_dim   = embed_dim

        # Shared MLP: scalar params → dense embedding
        # param_dim=12 → 128 → 128  (tiny)
        self.mlp = nn.Sequential(
            nn.Linear(param_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        # Per-scale: embedding → small spatial field
        # embed_dim → scale_width * BASE_NX * BASE_NT
        base_flat = self.BASE_NX * self.BASE_NT  # 128
        self.scale_heads = nn.ModuleList([
            nn.Linear(embed_dim, scale_width * base_flat)
            for _ in range(n_scales)
        ])

        # Per-scale: lightweight conv refinement at full resolution
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(scale_width, scale_width, kernel_size=3, padding=1),
                nn.GELU(),
            )
            for _ in range(n_scales)
        ])

    def forward(self, x_vec: torch.Tensor) -> torch.Tensor:
        B = x_vec.shape[0]

        # (B, embed_dim)
        emb = self.mlp(x_vec)

        scale_fields = []
        for k in range(self.n_scales):
            # (B, scale_width * BASE_NX * BASE_NT)
            field_k = self.scale_heads[k](emb)
            # (B, scale_width, BASE_NX, BASE_NT)
            field_k = field_k.view(B, self.scale_width, self.BASE_NX, self.BASE_NT)
            # upsample to full resolution — once per scale
            field_k = F.interpolate(
                field_k,
                size=(self.Nx, self.Nt),
                mode="bilinear",
                align_corners=False
            )
            # lightweight refinement conv at full resolution
            field_k = self.scale_convs[k](field_k)
            scale_fields.append(field_k)

        # (B, n_scales * scale_width, Nx, Nt)
        return torch.cat(scale_fields, dim=1)


class CrossScaleFusion(nn.Module):
    def __init__(self, in_ch: int, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, width, kernel_size=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.norm  = nn.GroupNorm(num_groups=min(8, width), num_channels=width)

        hidden = max(width // 4, 4)
        self.se_fc1 = nn.Linear(width, hidden)
        self.se_fc2 = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = F.gelu(self.conv1(x))
        x  = F.gelu(self.norm(self.conv2(x)))
        se = x.mean(dim=(2, 3))
        se = F.gelu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        return x * se.unsqueeze(-1).unsqueeze(-1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes_x: int, modes_t: int):
        super().__init__()
        self.modes_x = modes_x
        self.modes_t = modes_t
        scale = 1.0 / (in_ch * out_ch) ** 0.5          # fixed: sqrt not product
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, nx, nt = x.shape
        x_ft = rfft2(x, s=(nx, nt), norm="ortho")       # fixed: ortho = energy-preserving

        mx = min(self.modes_x, x_ft.shape[2])
        mt = min(self.modes_t, x_ft.shape[3])

        out_ft = torch.zeros(
            B, self.weight.shape[1], nx, nt // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :mx, :mt] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :mx, :mt],
            self.weight[:, :, :mx, :mt]
        )

        return irfft2(out_ft, s=(nx, nt), norm="ortho")  # fixed: ortho


class MIFNOBlock(nn.Module):
    def __init__(self, ch: int, cond_ch: int, modes_x: int, modes_t: int,
                 dropout: float = 0.1):
        super().__init__()
        self.spec     = SpectralConv2d(ch, ch, modes_x, modes_t)
        self.lin      = nn.Conv2d(ch, ch, kernel_size=1)
        self.norm     = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.drop     = nn.Dropout2d(p=dropout)

        # FiLM conditioning: learn per-channel scale and shift from cond
        # much more expressive than the original additive cond_proj
        self.film_fc  = nn.Conv2d(cond_ch, 2 * ch, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        film        = self.film_fc(cond)
        gamma, beta = film.chunk(2, dim=1)              # each (B, ch, Nx, Nt)
        h = self.spec(x) + self.lin(x)
        h = self.norm(h)
        h = h * (1 + gamma) + beta                      # FiLM modulation
        h = self.drop(F.gelu(h))
        return h


class MIFNO(nn.Module):
    def __init__(
        self,
        Nx: int,
        Nt: int,
        param_dim: int,
        n_scales: int,
        scale_width: int,
        width: int,
        depth: int,
        modes_x: int,
        modes_t: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.param_dim = param_dim

        # Single batched encoder replaces param_dim separate encoders
        self.encoder = BatchedMultiscaleEncoder(
            Nx, Nt, param_dim, n_scales, scale_width, embed_dim=128
        )

        in_ch = n_scales * scale_width          # e.g. 3 * 16 = 48  (NOT * param_dim)
        self.fusion = CrossScaleFusion(in_ch, width)

        self.blocks = nn.ModuleList([
            MIFNOBlock(width, width, modes_x, modes_t, dropout=dropout)
            for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 2, kernel_size=1),
        )

    def forward(self, x_vec: torch.Tensor) -> torch.Tensor:
        # One batched encoder call instead of param_dim sequential calls
        multi_input = self.encoder(x_vec)       # (B, n_scales*param_dim*scale_width, Nx, Nt)
        cond = self.fusion(multi_input)         # (B, width, Nx, Nt)

        x = cond
        for blk in self.blocks:
            x = blk(x, cond)                   # FiLM conditioning every block

        return self.head(x)                     # (B, 2, Nx, Nt)


# ============================================================
# Dataloaders
# ============================================================
X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train_combined, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
Y_val_t   = torch.tensor(Y_val_combined,   dtype=torch.float32)

# pin_memory only helps for CUDA
pin = (device.type == "cuda")

train_loader = DataLoader(
    TensorDataset(X_train_t, Y_train_t),
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=args.num_workers,
    pin_memory=pin,
    persistent_workers=(args.num_workers > 0),
)

val_loader = DataLoader(
    TensorDataset(X_val_t, Y_val_t),
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers,
    pin_memory=pin,
    persistent_workers=(args.num_workers > 0),
)


# ============================================================
# Training setup
# ============================================================
model = MIFNO(
    Nx=Nx,
    Nt=Nt,
    param_dim=param_dim,
    n_scales=args.n_scales,
    scale_width=args.scale_width,
    width=args.width,
    depth=args.depth,
    modes_x=args.modes_x,
    modes_t=args.modes_t,
    dropout=args.dropout,
).to(device)

# torch.compile: ~20-50% speedup on PyTorch 2.x with CUDA
if args.compile:
    if hasattr(torch, "compile"):
        print("   Compiling model with torch.compile ...")
        model = torch.compile(model)
    else:
        print("   ⚠️  torch.compile not available (requires PyTorch 2.x), skipping.")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model created | trainable params: {total_params:,}")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.8,
    patience=80,
    threshold=1e-4,
    min_lr=1e-6,
)

cfg = {
    "Nx": Nx,
    "Nt": Nt,
    "param_dim": param_dim,
    "param_names": param_names,
    "n_scales": args.n_scales,
    "scale_width": args.scale_width,
    "width": args.width,
    "depth": args.depth,
    "modes_x": args.modes_x,
    "modes_t": args.modes_t,
    "dropout": args.dropout,
    "T_mean": float(T_mean),
    "T_std":  float(T_std),
    "S_mean": float(S_mean),
    "S_std":  float(S_std),
    "results_dir": results_dir,
    "save_dir": save_dir,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "val_ratio": args.val_ratio,
    "seed": args.seed,
    "run_id": run_id,
    "base_name": base_name,
    "best_weights": save_path_best,
    "last_weights": save_path_last,
}
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=4)
shutil.copyfile(cfg_path, latest_cfg_path)
print("✅ Saved config")


# ============================================================
# Loss helpers
# ============================================================
def gradient_x(u: torch.Tensor) -> torch.Tensor:
    """Finite difference along thickness (x) direction. u: (B, Nx, Nt)"""
    return u[:, 1:, :] - u[:, :-1, :]


def compute_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    pred, target: (B, 2, Nx, Nt)
    Returns scalar loss and dict of components.
    """
    loss_T = F.mse_loss(pred[:, 0], target[:, 0])

    loss_S = F.smooth_l1_loss(
        pred[:, 1].contiguous(),
        target[:, 1].contiguous(),
        beta=0.5
    )

    loss_S_energy = F.mse_loss(
        pred[:, 1].abs().mean(dim=(1, 2)),
        target[:, 1].abs().mean(dim=(1, 2))
    )

    loss_grad_S = F.mse_loss(
        gradient_x(pred[:, 1]),
        gradient_x(target[:, 1])
    )

    total = 0.4 * loss_T + 1.2 * loss_S + 0.1 * loss_S_energy + 0.2 * loss_grad_S

    return total, {
        "loss_T": loss_T.item(),
        "loss_S": loss_S.item(),
        "loss_S_energy": loss_S_energy.item(),
    }


def evaluate(model, loader, device):
    model.eval()
    totals = {"loss": 0.0, "loss_T": 0.0, "loss_S": 0.0, "loss_S_energy": 0.0}
    count  = 0

    with torch.inference_mode():
        for xb, yb in loader:
            xb  = xb.to(device, non_blocking=True)
            yb  = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss, components = compute_loss(pred, yb)
            bs = xb.size(0)
            totals["loss"]          += loss.item() * bs
            totals["loss_T"]        += components["loss_T"] * bs
            totals["loss_S"]        += components["loss_S"] * bs
            totals["loss_S_energy"] += components["loss_S_energy"] * bs
            count += bs

    return {k: v / count for k, v in totals.items()}


# ============================================================
# Training loop
# ============================================================
history = {
    "run_id": run_id,
    "base_name": base_name,
    "train_loss": [],
    "val_loss": [],
    "train_loss_T": [],
    "val_loss_T": [],
    "train_loss_S": [],
    "val_loss_S": [],
    "lr": [],
}

best_val      = float("inf")
no_improve    = 0
early_stop_patience = 200   # stop if val doesn't improve for this many epochs

print(f"\nStarting training | epochs={args.epochs} | batch={args.batch_size} | "
      f"Nx={Nx} | Nt={Nt}\n")
t0 = time.time()

for epoch in range(1, args.epochs + 1):
    model.train()

    running = {"loss": 0.0, "loss_T": 0.0, "loss_S": 0.0, "loss_S_energy": 0.0}
    count   = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss, components = compute_loss(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = xb.size(0)
        running["loss"]          += loss.item() * bs
        running["loss_T"]        += components["loss_T"] * bs
        running["loss_S"]        += components["loss_S"] * bs
        running["loss_S_energy"] += components["loss_S_energy"] * bs
        count += bs

    train_metrics = {k: v / count for k, v in running.items()}
    val_metrics   = evaluate(model, val_loader, device)

    scheduler.step(val_metrics["loss"])
    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_metrics["loss"])
    history["val_loss"].append(val_metrics["loss"])
    history["train_loss_T"].append(train_metrics["loss_T"])
    history["val_loss_T"].append(val_metrics["loss_T"])
    history["train_loss_S"].append(train_metrics["loss_S"])
    history["val_loss_S"].append(val_metrics["loss_S"])
    history["lr"].append(current_lr)

    # Best model checkpoint
    if val_metrics["loss"] < best_val:
        best_val   = val_metrics["loss"]
        no_improve = 0
        torch.save(model.state_dict(), save_path_best)
        shutil.copyfile(save_path_best, latest_best_path)
    else:
        no_improve += 1

    # Periodic logging
    if epoch % args.save_every == 0 or epoch == 1:
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:4d} | "
            f"Train={train_metrics['loss']:.4e} "
            f"(T={train_metrics['loss_T']:.4e}, S={train_metrics['loss_S']:.4e}) | "
            f"Val={val_metrics['loss']:.4e} "
            f"(T={val_metrics['loss_T']:.4e}, S={val_metrics['loss_S']:.4e}) | "
            f"lr={current_lr:.2e} | {elapsed:.0f}s elapsed"
        )
        # Save history only on log epochs (not every epoch)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
        shutil.copyfile(history_path, latest_history_path)

    # Early stopping
    if no_improve >= early_stop_patience:
        print(f"\n⏹️  Early stopping at epoch {epoch} "
              f"(no val improvement for {early_stop_patience} epochs)")
        break

# Final saves
torch.save(model.state_dict(), save_path_last)
shutil.copyfile(save_path_last, latest_last_path)

# Final history flush
with open(history_path, "w") as f:
    json.dump(history, f, indent=4)
shutil.copyfile(history_path, latest_history_path)

elapsed = time.time() - t0
print(f"\n⏱️  Training time: {elapsed:.2f}s")
print(f"✅ Saved LAST weights → {save_path_last}")
print(f"🏁 Saved BEST weights → {save_path_best} (best_val={best_val:.6e})")
print(f"✅ Saved history      → {history_path}")
print(f"✅ Saved config       → {cfg_path}")

print("\n📦 Latest pointers:")
print(f"   latest best   → {latest_best_path}")
print(f"   latest last   → {latest_last_path}")
print(f"   latest config → {latest_cfg_path}")
print(f"   latest hist   → {latest_history_path}")