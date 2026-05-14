"""
compare_fem_mifno_mionet.py

Compares FEM ground truth against MIFNO and MIONet predictions.
Model architectures and scaler filenames match exactly:
  - train_MIFNO_multizone.py  → MIFNO dir
  - train_MIONET.py           → MIONET dir
"""
import os
import re
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.fft import rfft2, irfft2


# ============================================================
# Device
# ============================================================
def pick_device(req: str) -> torch.device:
    req = req.lower().strip()
    if req == "cpu":
        return torch.device("cpu")
    if req in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA not available")
    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS not available")
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError("device must be auto|cpu|cuda|mps")


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--device",   type=str,   default="cpu")
parser.add_argument("--split",    type=str,   default="train")
parser.add_argument("--case_id",  type=int,   default=0)
parser.add_argument("--t_idx",    type=int,   default=3299)
parser.add_argument("--dt",       type=float, default=0.1)
parser.add_argument("--velocity", type=float, default=0.16417)
parser.add_argument("--mifno_dir",  type=str, default="MIFNO")
parser.add_argument("--mionet_dir", type=str, default="MIONET")
args = parser.parse_args()

device = pick_device(args.device)
print(f"Comparison | device={device}")


# ============================================================
# Helpers
# ============================================================
def extract_case_idx(fname: str) -> int:
    m = re.search(r"case[_]?(\d+)|case(\d+)", fname)
    if m:
        for g in m.groups():
            if g is not None:
                return int(g)
    return 10**9


def rel_l2(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)


ZONE_TIME_WINDOWS = {
    "A1": (0.0,    54.7),
    "A2": (54.7,  109.6),
    "B1": (109.6, 182.6),
    "B2": (182.6, 255.57),
    "C1": (255.57, 328.5),
}


def add_zone_lines_time(ax, t_axis):
    for zone, (t0, t1) in ZONE_TIME_WINDOWS.items():
        if t0 < t_axis[-1]:
            center = 0.5 * (max(t0, t_axis[0]) + min(t1, t_axis[-1]))
            ax.text(center, 1.02, zone,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9)
        if t_axis[0] <= t1 <= t_axis[-1]:
            ax.axvline(t1, linestyle="--", linewidth=0.8, color="gray")


def add_zone_lines_distance(ax, x_axis, velocity):
    for zone, (t0, t1) in ZONE_TIME_WINDOWS.items():
        x0, x1 = velocity * t0, velocity * t1
        if x0 < x_axis[-1]:
            center = 0.5 * (max(x0, x_axis[0]) + min(x1, x_axis[-1]))
            ax.text(center, 1.02, zone,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9)
        if x_axis[0] <= x1 <= x_axis[-1]:
            ax.axvline(x1, linestyle="--", linewidth=0.8, color="gray")


def savefig(path):
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# MIFNO — exact copy from predict_MIFNO_multi-zone.py
# ============================================================
class BatchedMultiscaleEncoder(nn.Module):
    BASE_NX = 8
    BASE_NT = 16

    def __init__(self, Nx, Nt, param_dim, n_scales, scale_width, embed_dim=128):
        super().__init__()
        self.Nx, self.Nt = Nx, Nt
        self.param_dim, self.n_scales, self.scale_width = param_dim, n_scales, scale_width
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(param_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
        )
        base_flat = self.BASE_NX * self.BASE_NT
        self.scale_heads = nn.ModuleList([
            nn.Linear(embed_dim, scale_width * base_flat) for _ in range(n_scales)
        ])
        self.scale_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(scale_width, scale_width, kernel_size=3, padding=1), nn.GELU())
            for _ in range(n_scales)
        ])

    def forward(self, x_vec):
        B = x_vec.shape[0]
        emb = self.mlp(x_vec)
        scale_fields = []
        for k in range(self.n_scales):
            field_k = self.scale_heads[k](emb).view(B, self.scale_width, self.BASE_NX, self.BASE_NT)
            field_k = F.interpolate(field_k, size=(self.Nx, self.Nt), mode="bilinear", align_corners=False)
            field_k = self.scale_convs[k](field_k)
            scale_fields.append(field_k)
        return torch.cat(scale_fields, dim=1)


class CrossScaleFusion(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, width, kernel_size=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.norm  = nn.GroupNorm(num_groups=min(8, width), num_channels=width)
        hidden = max(width // 4, 4)
        self.se_fc1 = nn.Linear(width, hidden)
        self.se_fc2 = nn.Linear(hidden, width)

    def forward(self, x):
        x  = F.gelu(self.conv1(x))
        x  = F.gelu(self.norm(self.conv2(x)))
        se = x.mean(dim=(2, 3))
        se = F.gelu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        return x * se.unsqueeze(-1).unsqueeze(-1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes_x, modes_t):
        super().__init__()
        self.modes_x, self.modes_t = modes_x, modes_t
        scale = 1.0 / (in_ch * out_ch) ** 0.5
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat))

    def forward(self, x):
        B, C, nx, nt = x.shape
        x_ft = rfft2(x, s=(nx, nt), norm="ortho")
        mx = min(self.modes_x, x_ft.shape[2])
        mt = min(self.modes_t, x_ft.shape[3])
        out_ft = torch.zeros(B, self.weight.shape[1], nx, nt // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :mx, :mt] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :mx, :mt], self.weight[:, :, :mx, :mt])
        return irfft2(out_ft, s=(nx, nt), norm="ortho")


class MIFNOBlock(nn.Module):
    def __init__(self, ch, cond_ch, modes_x, modes_t, dropout=0.1):
        super().__init__()
        self.spec    = SpectralConv2d(ch, ch, modes_x, modes_t)
        self.lin     = nn.Conv2d(ch, ch, kernel_size=1)
        self.norm    = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.drop    = nn.Dropout2d(p=dropout)
        self.film_fc = nn.Conv2d(cond_ch, 2 * ch, kernel_size=1)

    def forward(self, x, cond):
        film        = self.film_fc(cond)
        gamma, beta = film.chunk(2, dim=1)
        h = self.spec(x) + self.lin(x)
        h = self.norm(h)
        h = h * (1 + gamma) + beta
        return self.drop(F.gelu(h))


class MIFNO(nn.Module):
    def __init__(self, Nx, Nt, param_dim, n_scales, scale_width,
                 width, depth, modes_x, modes_t, dropout=0.1):
        super().__init__()
        self.encoder = BatchedMultiscaleEncoder(Nx, Nt, param_dim, n_scales, scale_width)
        in_ch = n_scales * scale_width
        self.fusion = CrossScaleFusion(in_ch, width)
        self.blocks = nn.ModuleList([
            MIFNOBlock(width, width, modes_x, modes_t, dropout=dropout)
            for _ in range(depth)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(width // 2, 2, kernel_size=1),
        )

    def forward(self, x_vec):
        multi_input = self.encoder(x_vec)
        cond = self.fusion(multi_input)
        x = cond
        for blk in self.blocks:
            x = blk(x, cond)
        return self.head(x)


# ============================================================
# MIONet — exact copy from predict_MIONET.py / train_MIONET.py
# ============================================================
def make_mlp(in_dim, hidden_dim, out_dim, depth):
    layers, d0 = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d0, hidden_dim), nn.GELU()]
        d0 = hidden_dim
    layers += [nn.Linear(d0, out_dim)]
    return nn.Sequential(*layers)


class MIONet(nn.Module):
    def __init__(self, param_dim, latent_dim, branch_width, branch_depth,
                 trunk_width, trunk_depth, Nx, Nt):
        super().__init__()
        self.Nx, self.Nt, self.latent_dim = Nx, Nt, latent_dim
        self.branch    = make_mlp(param_dim, branch_width, 2 * latent_dim, branch_depth)
        self.trunk     = make_mlp(2, trunk_width, 2 * latent_dim, trunk_depth)
        self.bias_head = nn.Parameter(torch.zeros(2))

    def forward(self, params, coords):
        B, P = params.shape[0], coords.shape[0]
        branch_out = self.branch(params).view(B, 2, self.latent_dim)
        trunk_out  = self.trunk(coords).view(P, 2, self.latent_dim)
        out = torch.einsum("bcl,pcl->bcp", branch_out, trunk_out) + self.bias_head.view(1, 2, 1)
        return out.view(B, 2, self.Nx, self.Nt)


# ============================================================
# Load MIFNO
# ============================================================
mifno_dir    = args.mifno_dir
cfg_path     = os.path.join(mifno_dir, "latest_config.json")
weights_path = os.path.join(mifno_dir, "latest_model_best.pt")

for p in (cfg_path, weights_path):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")

with open(cfg_path) as f:
    mifno_cfg = json.load(f)

Nx        = mifno_cfg["Nx"]
Nt        = mifno_cfg["Nt"]
param_dim = mifno_cfg["param_dim"]

# Scalers — global scalars (train_MIFNO saves scalar mean/std)
x_mean_mifno  = np.load(os.path.join(mifno_dir, "x_scaler_mean_mifno_mz.npy"))
x_std_mifno   = np.load(os.path.join(mifno_dir, "x_scaler_scale_mifno_mz.npy"))
T_mean_mifno  = float(np.load(os.path.join(mifno_dir, "y_temp_mean_mz.npy")).flat[0])
T_std_mifno   = float(np.load(os.path.join(mifno_dir, "y_temp_std_mz.npy")).flat[0])
S_mean_mifno  = float(np.load(os.path.join(mifno_dir, "y_stress_mean_mz.npy")).flat[0])
S_std_mifno   = float(np.load(os.path.join(mifno_dir, "y_stress_std_mz.npy")).flat[0])

mifno_model = MIFNO(
    Nx=Nx, Nt=Nt, param_dim=param_dim,
    n_scales    = mifno_cfg["n_scales"],
    scale_width = mifno_cfg["scale_width"],
    width       = mifno_cfg["width"],
    depth       = mifno_cfg["depth"],
    modes_x     = mifno_cfg["modes_x"],
    modes_t     = mifno_cfg["modes_t"],
    dropout     = mifno_cfg.get("dropout", 0.1),
).to(device)

mifno_model.load_state_dict(torch.load(weights_path, map_location=device), strict=True)
mifno_model.eval()
print(f"  MIFNO loaded  (Nx={Nx}, Nt={Nt}, param_dim={param_dim})")


# ============================================================
# Load MIONet
# ============================================================
mionet_dir    = args.mionet_dir
mcfg_path     = os.path.join(mionet_dir, "latest_config_mionet.json")
mweights_path = os.path.join(mionet_dir, "latest_model_best_mionet.pt")

for p in (mcfg_path, mweights_path):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")

with open(mcfg_path) as f:
    mionet_cfg = json.load(f)

if mionet_cfg["Nx"] != Nx or mionet_cfg["Nt"] != Nt or mionet_cfg["param_dim"] != param_dim:
    raise ValueError(
        f"MIFNO and MIONet grid mismatch: "
        f"MIFNO=({Nx},{Nt},{param_dim}) vs MIONet=({mionet_cfg['Nx']},{mionet_cfg['Nt']},{mionet_cfg['param_dim']})"
    )

# Scalers — per-pixel arrays (train_MIONET uses StandardScaler flattened)
x_mean_mionet        = np.load(os.path.join(mionet_dir, "x_scaler_mean_mionet_mz.npy"))
x_std_mionet         = np.load(os.path.join(mionet_dir, "x_scaler_scale_mionet_mz.npy"))
y_temp_mean_mionet   = np.load(os.path.join(mionet_dir, "y_temp_scaler_mean_mionet_mz.npy"))
y_temp_std_mionet    = np.load(os.path.join(mionet_dir, "y_temp_scaler_scale_mionet_mz.npy"))
y_stress_mean_mionet = np.load(os.path.join(mionet_dir, "y_stress_scaler_mean_mionet_mz.npy"))
y_stress_std_mionet  = np.load(os.path.join(mionet_dir, "y_stress_scaler_scale_mionet_mz.npy"))

mionet_model = MIONet(
    param_dim    = param_dim,
    latent_dim   = mionet_cfg["latent_dim"],
    branch_width = mionet_cfg["branch_width"],
    branch_depth = mionet_cfg["branch_depth"],
    trunk_width  = mionet_cfg["trunk_width"],
    trunk_depth  = mionet_cfg["trunk_depth"],
    Nx=Nx, Nt=Nt,
).to(device)

mionet_model.load_state_dict(torch.load(mweights_path, map_location=device), strict=True)
mionet_model.eval()
print(f"  MIONet loaded")


# ============================================================
# Load FEM dataset
# ============================================================
data_dir = os.path.join("results", args.split)

param_files  = sorted([f for f in os.listdir(data_dir) if f.startswith("params_case_")],
                       key=extract_case_idx)
temp_files   = sorted([f for f in os.listdir(data_dir) if f.startswith("temperature_all_case")],
                       key=extract_case_idx)
stress_files = sorted([f for f in os.listdir(data_dir) if f.startswith("stress_all_case")],
                       key=extract_case_idx)

X_list, T_list, S_list = [], [], []
for p, t, s in zip(param_files, temp_files, stress_files):
    X_list.append(np.loadtxt(os.path.join(data_dir, p), delimiter=",", skiprows=1))
    T_list.append(np.loadtxt(os.path.join(data_dir, t)))
    S_list.append(np.loadtxt(os.path.join(data_dir, s)))

X        = np.array(X_list, dtype=np.float32)
Y_temp   = np.array(T_list).reshape(len(X), Nt, Nx).transpose(0, 2, 1).astype(np.float32)
Y_stress = np.array(S_list).reshape(len(X), Nt, Nx).transpose(0, 2, 1).astype(np.float32)

N = len(X)
print(f"  Dataset: N={N}  split={args.split}")


# ============================================================
# Inference
# ============================================================
X_sc_mifno  = ((X - x_mean_mifno)  / x_std_mifno).astype(np.float32)
X_sc_mionet = ((X - x_mean_mionet) / x_std_mionet).astype(np.float32)

# MIONet coordinate grid
x_coords = np.linspace(0.0, 1.0, Nx, dtype=np.float32)
t_coords = np.linspace(0.0, 1.0, Nt, dtype=np.float32)
XX, TT   = np.meshgrid(x_coords, t_coords, indexing="ij")
coords   = np.stack([XX.ravel(), TT.ravel()], axis=1).astype(np.float32)
coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

# Warm-up
with torch.no_grad():
    _ = mifno_model(torch.tensor(X_sc_mifno[:1],  device=device))
    _ = mionet_model(torch.tensor(X_sc_mionet[:1], device=device), coords_t)

# MIFNO inference
sync(device)
t0 = time.time()
with torch.no_grad():
    pred_mifno = mifno_model(torch.tensor(X_sc_mifno, device=device)).cpu().numpy()
sync(device)
mifno_time = time.time() - t0

# MIONet inference
sync(device)
t1 = time.time()
with torch.no_grad():
    pred_mionet = mionet_model(torch.tensor(X_sc_mionet, device=device), coords_t).cpu().numpy()
sync(device)
mionet_time = time.time() - t1

# Inverse scale
Y_pred_temp_mifno    = pred_mifno[:,  0] * T_std_mifno   + T_mean_mifno
Y_pred_stress_mifno  = pred_mifno[:,  1] * S_std_mifno   + S_mean_mifno
Y_pred_temp_mionet   = pred_mionet[:, 0] * y_temp_std_mionet   + y_temp_mean_mionet
Y_pred_stress_mionet = pred_mionet[:, 1] * y_stress_std_mionet + y_stress_mean_mionet


# ============================================================
# Metrics
# ============================================================
print("\nResults")
print(f"  MIFNO  | L2 Temp={rel_l2(Y_pred_temp_mifno,   Y_temp)*100:.3f}%  "
      f"L2 Stress={rel_l2(Y_pred_stress_mifno,  Y_stress)*100:.3f}%  "
      f"({mifno_time/N*1000:.2f} ms/sample)")
print(f"  MIONet | L2 Temp={rel_l2(Y_pred_temp_mionet,  Y_temp)*100:.3f}%  "
      f"L2 Stress={rel_l2(Y_pred_stress_mionet, Y_stress)*100:.3f}%  "
      f"({mionet_time/N*1000:.2f} ms/sample)")


# ============================================================
# Plots
# ============================================================
os.makedirs("plots_case_checks", exist_ok=True)

case_id = min(max(0, args.case_id), N - 1)
t_idx   = min(max(0, args.t_idx),   Nt - 1)
dt      = args.dt
vel     = args.velocity

t_axis = (np.arange(Nt) + 1) * dt          # t=0.1 ... 330.0
x_axis = vel * t_axis

surf = 0
mid  = Nx // 2

# 1) Surface temperature vs time
plt.figure(figsize=(10, 5))
plt.plot(t_axis, Y_temp[case_id, surf],             "--",  label="FEM")
plt.plot(t_axis, Y_pred_temp_mifno[case_id, surf],         label="MIFNO")
plt.plot(t_axis, Y_pred_temp_mionet[case_id, surf],        label="MIONet")
plt.xlabel("Time (s)"); plt.ylabel("Temperature (K)")
plt.title("Surface temperature vs time", pad=18)
plt.legend(); plt.grid(True)
add_zone_lines_time(plt.gca(), t_axis)
plt.tight_layout()
savefig(f"plots_case_checks/compare_surface_temp_vs_time_case{case_id}.png")

# 2) Surface temperature vs distance
plt.figure(figsize=(10, 5))
plt.plot(x_axis, Y_temp[case_id, surf],             "--",  label="FEM")
plt.plot(x_axis, Y_pred_temp_mifno[case_id, surf],         label="MIFNO")
plt.plot(x_axis, Y_pred_temp_mionet[case_id, surf],        label="MIONet")
plt.xlabel("Lehr distance (m)"); plt.ylabel("Temperature (K)")
plt.title("Surface temperature vs distance", pad=18)
plt.legend(); plt.grid(True)
add_zone_lines_distance(plt.gca(), x_axis, vel)
plt.tight_layout()
savefig(f"plots_case_checks/compare_surface_temp_vs_distance_case{case_id}.png")

# 3) Stress vs time
plt.figure(figsize=(10, 5))
plt.plot(t_axis, Y_stress[case_id, surf]/1e6,            "--", label="FEM surface")
plt.plot(t_axis, Y_pred_stress_mifno[case_id, surf]/1e6,       label="MIFNO surface")
plt.plot(t_axis, Y_pred_stress_mionet[case_id, surf]/1e6,      label="MIONet surface")
plt.plot(t_axis, Y_stress[case_id, mid]/1e6,             "--", label="FEM mid")
plt.plot(t_axis, Y_pred_stress_mifno[case_id, mid]/1e6,        label="MIFNO mid")
plt.plot(t_axis, Y_pred_stress_mionet[case_id, mid]/1e6,       label="MIONet mid")
plt.xlabel("Time (s)"); plt.ylabel("Stress (MPa)")
plt.title("Stress vs time", pad=18)
plt.legend(ncol=2); plt.grid(True)
add_zone_lines_time(plt.gca(), t_axis)
plt.tight_layout()
savefig(f"plots_case_checks/compare_stress_vs_time_case{case_id}.png")

# 4) Stress vs distance
plt.figure(figsize=(10, 5))
plt.plot(x_axis, Y_stress[case_id, surf]/1e6,            "--", label="FEM surface")
plt.plot(x_axis, Y_pred_stress_mifno[case_id, surf]/1e6,       label="MIFNO surface")
plt.plot(x_axis, Y_pred_stress_mionet[case_id, surf]/1e6,      label="MIONet surface")
plt.plot(x_axis, Y_stress[case_id, mid]/1e6,             "--", label="FEM mid")
plt.plot(x_axis, Y_pred_stress_mifno[case_id, mid]/1e6,        label="MIFNO mid")
plt.plot(x_axis, Y_pred_stress_mionet[case_id, mid]/1e6,       label="MIONet mid")
plt.xlabel("Lehr distance (m)"); plt.ylabel("Stress (MPa)")
plt.title("Stress vs distance", pad=18)
plt.legend(ncol=2); plt.grid(True)
add_zone_lines_distance(plt.gca(), x_axis, vel)
plt.tight_layout()
savefig(f"plots_case_checks/compare_stress_vs_distance_case{case_id}.png")

# 5) Temperature profile through thickness at t_idx
t_phys = t_axis[t_idx]
z_axis = np.linspace(-2.0, 2.0, Nx)
plt.figure(figsize=(9, 5))
plt.plot(z_axis, Y_temp[case_id, :, t_idx],            "--", label="FEM")
plt.plot(z_axis, Y_pred_temp_mifno[case_id, :, t_idx],       label="MIFNO")
plt.plot(z_axis, Y_pred_temp_mionet[case_id, :, t_idx],      label="MIONet")
plt.xlabel("z (mm)"); plt.ylabel("Temperature (K)")
plt.title(f"Temperature profile at t={t_phys:.1f}s")
plt.legend(); plt.grid(True)
plt.tight_layout()
savefig(f"plots_case_checks/compare_temp_thickness_case{case_id}_t{t_idx}.png")

# 6) Stress profile through thickness at t_idx
plt.figure(figsize=(9, 5))
plt.plot(z_axis, Y_stress[case_id, :, t_idx]/1e6,            "--", label="FEM")
plt.plot(z_axis, Y_pred_stress_mifno[case_id, :, t_idx]/1e6,       label="MIFNO")
plt.plot(z_axis, Y_pred_stress_mionet[case_id, :, t_idx]/1e6,      label="MIONet")
plt.xlabel("z (mm)"); plt.ylabel("Stress (MPa)")
plt.title(f"Stress profile at t={t_phys:.1f}s")
plt.legend(); plt.grid(True)
plt.tight_layout()
savefig(f"plots_case_checks/compare_stress_thickness_case{case_id}_t{t_idx}.png")

# 7) Field maps (2×3 grid)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
pairs = [
    (Y_temp[case_id],            "FEM Temp (K)"),
    (Y_pred_temp_mifno[case_id], "MIFNO Temp (K)"),
    (Y_pred_temp_mionet[case_id],"MIONet Temp (K)"),
    (Y_stress[case_id]/1e6,            "FEM Stress (MPa)"),
    (Y_pred_stress_mifno[case_id]/1e6, "MIFNO Stress (MPa)"),
    (Y_pred_stress_mionet[case_id]/1e6,"MIONet Stress (MPa)"),
]
for ax, (data, title) in zip(axes.ravel(), pairs):
    im = ax.imshow(data, aspect="auto", origin="lower",
                   extent=[t_axis[0], t_axis[-1], 0, Nx-1])
    ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel("Node index")
    fig.colorbar(im, ax=ax)
fig.tight_layout()
savefig(f"plots_case_checks/compare_field_maps_case{case_id}.png")

# 8) Error maps (2×2 grid)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
err_pairs = [
    (np.abs(Y_pred_temp_mifno[case_id]   - Y_temp[case_id]),   "MIFNO |Temp Error| (K)"),
    (np.abs(Y_pred_temp_mionet[case_id]  - Y_temp[case_id]),   "MIONet |Temp Error| (K)"),
    (np.abs(Y_pred_stress_mifno[case_id] - Y_stress[case_id])/1e6,  "MIFNO |Stress Error| (MPa)"),
    (np.abs(Y_pred_stress_mionet[case_id]- Y_stress[case_id])/1e6,  "MIONet |Stress Error| (MPa)"),
]
for ax, (data, title) in zip(axes.ravel(), err_pairs):
    im = ax.imshow(data, aspect="auto", origin="lower",
                   extent=[t_axis[0], t_axis[-1], 0, Nx-1])
    ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel("Node index")
    fig.colorbar(im, ax=ax)
fig.tight_layout()
savefig(f"plots_case_checks/compare_error_maps_case{case_id}.png")

# 9) Summary bar charts
fem_time_per_sample    = 9.7
mifno_time_per_sample  = mifno_time / N
mionet_time_per_sample = mionet_time / N
models = ["FEM", "MIFNO", "MIONet"]
temp_errs   = [0.0, rel_l2(Y_pred_temp_mifno,   Y_temp)*100,   rel_l2(Y_pred_temp_mionet,   Y_temp)*100]
stress_errs = [0.0, rel_l2(Y_pred_stress_mifno, Y_stress)*100, rel_l2(Y_pred_stress_mionet, Y_stress)*100]
times       = [fem_time_per_sample, mifno_time_per_sample, mionet_time_per_sample]
speedups    = [1.0, fem_time_per_sample/mifno_time_per_sample,
               fem_time_per_sample/mionet_time_per_sample]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, vals, title, ylabel in [
    (axes[0,0], temp_errs,   "Temperature accuracy", "L2 Error (%)"),
    (axes[0,1], stress_errs, "Stress accuracy",      "L2 Error (%)"),
    (axes[1,0], times,       "Time per sample",      "Time (s)"),
    (axes[1,1], speedups,    "Speed-up vs FEM",      "Speed-up (×)"),
]:
    bars = ax.bar(models, vals)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    for b, v in zip(bars, vals):
        label = f"{v:.2f}×" if "Speed" in title else f"{v:.3f}"
        ax.text(b.get_x() + b.get_width()/2,
                b.get_height() + 0.02 * max(max(vals), 1e-6),
                label, ha="center", va="bottom", fontsize=9)
fig.tight_layout()
savefig("plots_case_checks/compare_efficiency_summary.png")

print("\nAll plots saved to plots_case_checks/")