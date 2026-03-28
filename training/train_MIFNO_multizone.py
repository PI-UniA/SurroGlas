# train_MIFNO.py
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
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--val_ratio", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--width", type=int, default=96)
parser.add_argument("--depth", type=int, default=8)
parser.add_argument("--n_scales", type=int, default=3)
parser.add_argument("--scale_width", type=int, default=16)
parser.add_argument("--modes_x", type=int, default=12)
parser.add_argument("--modes_t", type=int, default=32)
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
    print("Using Apple MPS backend")


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


def infer_shape_from_flat(flat_len: int, n_space_candidates=None):
    if n_space_candidates is None:
        n_space_candidates = [20, 21, 49, 50, 100, 200, 500]

    for nx in n_space_candidates:
        if flat_len % nx == 0:
            nt = flat_len // nx
            return nt, nx

    divisors = [d for d in range(2, int(np.sqrt(flat_len)) + 1) if flat_len % d == 0]
    if not divisors:
        raise ValueError(f"Could not infer Nx/Nt from flattened length {flat_len}")

    nx = min(divisors, key=lambda x: abs(x - 20))
    nt = flat_len // nx
    return nt, nx


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
save_dir = args.save_dir
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

first_temp = np.loadtxt(os.path.join(results_dir, temp_files[0]))
Nt, Nx = infer_shape_from_flat(first_temp.size, n_space_candidates=[20, 21, 49, 50, 100, 200, 500])

X_params = []
Y_temperatures = []
Y_stresses = []

param_names = None

for p, t, s in zip(param_files, temp_files, stress_files):
    params, header = load_param_file(os.path.join(results_dir, p))
    temps = np.loadtxt(os.path.join(results_dir, t)).astype(np.float32)
    strs = np.loadtxt(os.path.join(results_dir, s)).astype(np.float32)

    if param_names is None:
        param_names = header
    else:
        if header != param_names:
            raise ValueError(f"Parameter header mismatch in {p}")

    if temps.size != Nt * Nx:
        raise ValueError(f"Temperature size mismatch in {t}: got {temps.size}, expected {Nt*Nx}")
    if strs.size != Nt * Nx:
        raise ValueError(f"Stress size mismatch in {s}: got {strs.size}, expected {Nt*Nx}")

    X_params.append(params)
    Y_temperatures.append(temps.reshape(Nt, Nx).T)  # (Nx, Nt)
    Y_stresses.append(strs.reshape(Nt, Nx).T)       # (Nx, Nt)

X = np.stack(X_params).astype(np.float32)
Y_temp = np.stack(Y_temperatures).astype(np.float32)
Y_strs = np.stack(Y_stresses).astype(np.float32)

N = X.shape[0]
param_dim = X.shape[1]

print("\n✅ Dataset loaded")
print(f"   results_dir = {results_dir}")
print(f"   N           = {N}")
print(f"   param_dim   = {param_dim}")
print(f"   Nx, Nt      = {Nx}, {Nt}")
print(f"   param_names = {param_names}")

# ============================================================
# Unique run naming
# ============================================================
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = f"mifno_multizone_Nx{Nx}_Nt{Nt}_N{N}_{run_id}"

save_path_last = os.path.join(save_dir, f"{base_name}_last.pt")
save_path_best = os.path.join(save_dir, f"{base_name}_best.pt")
history_path = os.path.join(save_dir, f"{base_name}_history.json")
cfg_path = os.path.join(save_dir, f"{base_name}_config.json")

latest_last_path = os.path.join(save_dir, "latest_model_last.pt")
latest_best_path = os.path.join(save_dir, "latest_model_best.pt")
latest_cfg_path = os.path.join(save_dir, "latest_config.json")
latest_history_path = os.path.join(save_dir, "latest_history.json")


# ============================================================
# Train/validation split
# ============================================================
idx_all = np.arange(N)
idx_train, idx_val = train_test_split(
    idx_all,
    test_size=args.val_ratio,
    random_state=args.seed,
    shuffle=True
)

X_train_raw = X[idx_train]
X_val_raw = X[idx_val]

Y_temp_train_raw = Y_temp[idx_train]
Y_temp_val_raw = Y_temp[idx_val]

Y_strs_train_raw = Y_strs[idx_train]
Y_strs_val_raw = Y_strs[idx_val]


# ============================================================
# Normalization
# ============================================================
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train_raw)
X_val = x_scaler.transform(X_val_raw)

yT_scaler = StandardScaler()
Y_temp_train = yT_scaler.fit_transform(Y_temp_train_raw.reshape(len(idx_train), -1)).reshape(len(idx_train), Nx, Nt)
Y_temp_val = yT_scaler.transform(Y_temp_val_raw.reshape(len(idx_val), -1)).reshape(len(idx_val), Nx, Nt)

yS_scaler = StandardScaler()
Y_strs_train = yS_scaler.fit_transform(Y_strs_train_raw.reshape(len(idx_train), -1)).reshape(len(idx_train), Nx, Nt)
Y_strs_val = yS_scaler.transform(Y_strs_val_raw.reshape(len(idx_val), -1)).reshape(len(idx_val), Nx, Nt)

Y_train_combined = np.stack([Y_temp_train, Y_strs_train], axis=1)
Y_val_combined = np.stack([Y_temp_val, Y_strs_val], axis=1)

# fixed scaler filenames for easy prediction
np.save(os.path.join(save_dir, "x_scaler_mean_mifno_mz.npy"), x_scaler.mean_)
np.save(os.path.join(save_dir, "x_scaler_scale_mifno_mz.npy"), x_scaler.scale_)
np.save(os.path.join(save_dir, "y_temp_scaler_mean_mz.npy"), yT_scaler.mean_.reshape(Nx, Nt))
np.save(os.path.join(save_dir, "y_temp_scaler_scale_mz.npy"), yT_scaler.scale_.reshape(Nx, Nt))
np.save(os.path.join(save_dir, "y_stress_scaler_mean_mz.npy"), yS_scaler.mean_.reshape(Nx, Nt))
np.save(os.path.join(save_dir, "y_stress_scaler_scale_mz.npy"), yS_scaler.scale_.reshape(Nx, Nt))

np.save(os.path.join(save_dir, "X_train_mifno_mz.npy"), X_train)
np.save(os.path.join(save_dir, "Y_train_mifno_combined_mz.npy"), Y_train_combined)
np.save(os.path.join(save_dir, "X_val_mifno_mz.npy"), X_val)
np.save(os.path.join(save_dir, "Y_val_mifno_combined_mz.npy"), Y_val_combined)

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

print("✅ Saved scalers and config")


# ============================================================
# Model
# ============================================================
class MultiscaleInputEncoder(nn.Module):
    def __init__(self, Nx: int, Nt: int, n_scales: int, scale_width: int):
        super().__init__()
        self.Nx = Nx
        self.Nt = Nt
        self.n_scales = n_scales
        self.scale_width = scale_width

        self.scale_lifters = nn.ModuleList()
        self.scale_convs = nn.ModuleList()

        for k in range(n_scales):
            nx_k = max(Nx // (2 ** k), 4)
            nt_k = max(Nt // (2 ** k), 4)

            self.scale_lifters.append(
                nn.Sequential(
                    nn.Linear(1, 64),
                    nn.GELU(),
                    nn.Linear(64, nx_k * nt_k),
                )
            )

            self.scale_convs.append(
                nn.Sequential(
                    nn.Conv2d(1, scale_width, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(scale_width, scale_width, kernel_size=3, padding=1),
                )
            )

    def forward(self, x_scalar: torch.Tensor) -> torch.Tensor:
        B = x_scalar.shape[0]
        scale_fields = []

        for k in range(self.n_scales):
            nx_k = max(self.Nx // (2 ** k), 4)
            nt_k = max(self.Nt // (2 ** k), 4)

            field_k = self.scale_lifters[k](x_scalar)
            field_k = field_k.view(B, 1, nx_k, nt_k)
            field_k = self.scale_convs[k](field_k)
            field_k = F.interpolate(
                field_k,
                size=(self.Nx, self.Nt),
                mode="bilinear",
                align_corners=False
            )
            scale_fields.append(field_k)

        return torch.cat(scale_fields, dim=1)


class CrossScaleFusion(nn.Module):
    def __init__(self, in_ch: int, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, width, kernel_size=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=8 if width >= 8 else 1, num_channels=width)

        hidden = max(width // 4, 4)
        self.se_fc1 = nn.Linear(width, hidden)
        self.se_fc2 = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.norm(self.conv2(x)))

        se = x.mean(dim=(2, 3))
        se = F.gelu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        x = x * se.unsqueeze(-1).unsqueeze(-1)

        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes_x: int, modes_t: int):
        super().__init__()
        self.modes_x = modes_x
        self.modes_t = modes_t
        self.weight = nn.Parameter(
            torch.randn(in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat) / (in_ch * out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, nx, nt = x.shape
        x_ft = rfft2(x, s=(nx, nt), norm="forward")

        mx = min(self.modes_x, x_ft.shape[2])
        mt = min(self.modes_t, x_ft.shape[3])

        out_ft = torch.zeros(
            B, self.weight.shape[1], nx, nt // 2 + 1,
            device=x.device,
            dtype=torch.cfloat
        )

        out_ft[:, :, :mx, :mt] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :mx, :mt],
            self.weight[:, :, :mx, :mt]
        )

        return irfft2(out_ft, s=(nx, nt), norm="forward")


class MIFNOBlock(nn.Module):
    def __init__(self, ch: int, cond_ch: int, modes_x: int, modes_t: int):
        super().__init__()
        self.spec = SpectralConv2d(ch, ch, modes_x, modes_t)
        self.lin = nn.Conv2d(ch, ch, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8 if ch >= 8 else 1, num_channels=ch)
        self.cond_proj = nn.Conv2d(cond_ch, ch, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spec(x) + self.lin(x) + self.cond_proj(cond)))


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
    ):
        super().__init__()
        self.param_dim = param_dim

        self.input_encoders = nn.ModuleList([
            MultiscaleInputEncoder(Nx, Nt, n_scales, scale_width)
            for _ in range(param_dim)
        ])

        in_ch = param_dim * n_scales * scale_width
        self.fusion = CrossScaleFusion(in_ch, width)

        self.blocks = nn.ModuleList([
            MIFNOBlock(width, width, modes_x, modes_t)
            for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 2, kernel_size=1),
        )

    def forward(self, x_vec: torch.Tensor) -> torch.Tensor:
        per_input_fields = [
            enc(x_vec[:, i:i+1])
            for i, enc in enumerate(self.input_encoders)
        ]
        multi_input = torch.cat(per_input_fields, dim=1)
        cond = self.fusion(multi_input)

        x = cond
        for blk in self.blocks:
            x = blk(x, cond)

        return self.head(x)


# ============================================================
# Dataloaders
# ============================================================
X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train_combined, dtype=torch.float32)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
Y_val_t = torch.tensor(Y_val_combined, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_t, Y_train_t),
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    TensorDataset(X_val_t, Y_val_t),
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False
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
).to(device)

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
    threshold=1e-4
)

def gradient_x(u):
    """
    u shape: (B, Nx, Nt)
    finite difference along thickness direction x
    output: (B, Nx-1, Nt)
    """
    return u[:, 1:, :] - u[:, :-1, :]

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_T = 0.0
    total_S = 0.0
    total_S_energy = 0.0
    total_count = 0

    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            loss_T = F.mse_loss(pred[:, 0], yb[:, 0])

            loss_S = F.smooth_l1_loss(
                pred[:, 1].contiguous(),
                yb[:, 1].contiguous(),
                beta=0.5
            )

            loss_S_energy = F.mse_loss(
                pred[:, 1].abs().mean(dim=(1, 2)),
                yb[:, 1].abs().mean(dim=(1, 2))
            )

            loss_grad_S = F.mse_loss(
                gradient_x(pred[:, 1]),
                gradient_x(yb[:, 1])
            )

            loss = 0.4 * loss_T + 1.2 * loss_S + 0.1 * loss_S_energy + 0.2 * loss_grad_S

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_T += loss_T.item() * bs
            total_S += loss_S.item() * bs
            total_S_energy += loss_S_energy.item() * bs
            total_count += bs

    return {
        "loss": total_loss / total_count,
        "loss_T": total_T / total_count,
        "loss_S": total_S / total_count,
        "loss_S_energy": total_S_energy / total_count,
    }


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

best_val = float("inf")

print(f"\nStarting training | epochs={args.epochs} | batch={args.batch_size}\n")
t0 = time.time()

for epoch in range(1, args.epochs + 1):
    model.train()

    running_loss = 0.0
    running_T = 0.0
    running_S = 0.0
    running_S_energy = 0.0
    count = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)

        pred = model(xb)

        loss_T = F.mse_loss(pred[:, 0], yb[:, 0])

        loss_S = F.smooth_l1_loss(
            pred[:, 1].contiguous(),
            yb[:, 1].contiguous(),
            beta=0.5
        )

        loss_S_energy = F.mse_loss(
            pred[:, 1].abs().mean(dim=(1, 2)),
            yb[:, 1].abs().mean(dim=(1, 2))
        )

        loss_grad_S = F.mse_loss(
            gradient_x(pred[:, 1]),
            gradient_x(yb[:, 1])
        )

        loss = 0.4 * loss_T + 1.2 * loss_S + 0.1 * loss_S_energy + 0.2 * loss_grad_S
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = xb.size(0)
        running_loss += loss.item() * bs
        running_T += loss_T.item() * bs
        running_S += loss_S.item() * bs
        running_S_energy += loss_S_energy.item() * bs
        count += bs

    train_metrics = {
        "loss": running_loss / count,
        "loss_T": running_T / count,
        "loss_S": running_S / count,
        "loss_S_energy": running_S_energy / count,
    }

    val_metrics = evaluate(model, val_loader, device)
    scheduler.step(val_metrics["loss"])

    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_metrics["loss"])
    history["val_loss"].append(val_metrics["loss"])
    history["train_loss_T"].append(train_metrics["loss_T"])
    history["val_loss_T"].append(val_metrics["loss_T"])
    history["train_loss_S"].append(train_metrics["loss_S"])
    history["val_loss_S"].append(val_metrics["loss_S"])
    history["lr"].append(current_lr)

    if val_metrics["loss"] < best_val:
        best_val = val_metrics["loss"]
        torch.save(model.state_dict(), save_path_best)
        shutil.copyfile(save_path_best, latest_best_path)

    if epoch % 25 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:4d} | "
            f"Train={train_metrics['loss']:.6e} "
            f"(T={train_metrics['loss_T']:.6e}, S={train_metrics['loss_S']:.6e}) | "
            f"Val={val_metrics['loss']:.6e} "
            f"(T={val_metrics['loss_T']:.6e}, S={val_metrics['loss_S']:.6e}) | "
            f"lr={current_lr:.2e}"
        )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    shutil.copyfile(history_path, latest_history_path)

torch.save(model.state_dict(), save_path_last)
shutil.copyfile(save_path_last, latest_last_path)

print(f"\n⏱️ Training time: {time.time() - t0:.2f}s")
print(f"✅ Saved LAST weights → {save_path_last}")
print(f"🏁 Saved BEST weights → {save_path_best} (best_val={best_val:.6e})")
print(f"✅ Saved history → {history_path}")
print(f"✅ Saved config  → {cfg_path}")

print("\n📦 Latest pointers:")
print(f"   latest best   → {latest_best_path}")
print(f"   latest last   → {latest_last_path}")
print(f"   latest config → {latest_cfg_path}")
print(f"   latest hist   → {latest_history_path}")