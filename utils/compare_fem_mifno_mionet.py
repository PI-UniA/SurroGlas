# compare_fem_mifno_mionet.py
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
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--split", type=str, default="test_unseen")
parser.add_argument("--case_id", type=int, default=0)
parser.add_argument("--t_idx", type=int, default=190)
parser.add_argument("--dt", type=float, default=0.1)
parser.add_argument("--velocity", type=float, default=0.24)
args = parser.parse_args()

device = pick_device(args.device)
print(f"🚀 Comparison | device={device}")


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


def gradient_x(u):
    return u[:, 1:, :] - u[:, :-1, :]


# ============================================================
# Shared plot helpers
# ============================================================
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
            ax.axvline(t1, linestyle="--", linewidth=1, color="gray")


def add_zone_lines_distance(ax, x_axis, velocity):
    for zone, (t0, t1) in ZONE_TIME_WINDOWS.items():
        x0 = velocity * t0
        x1 = velocity * t1
        if x0 < x_axis[-1]:
            center = 0.5 * (max(x0, x_axis[0]) + min(x1, x_axis[-1]))
            ax.text(center, 1.02, zone,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9)
        if x_axis[0] <= x1 <= x_axis[-1]:
            ax.axvline(x1, linestyle="--", linewidth=1, color="gray")


# ============================================================
# MIFNO model
# ============================================================
class MultiscaleInputEncoder(nn.Module):
    def __init__(self, Nx, Nt, n_scales, scale_width):
        super().__init__()
        self.Nx, self.Nt = Nx, Nt
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
                    nn.Conv2d(1, scale_width, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(scale_width, scale_width, 3, padding=1),
                )
            )

    def forward(self, x_scalar):
        B = x_scalar.shape[0]
        fields = []
        for k in range(self.n_scales):
            nx_k = max(self.Nx // (2 ** k), 4)
            nt_k = max(self.Nt // (2 ** k), 4)

            f = self.scale_lifters[k](x_scalar)
            f = f.view(B, 1, nx_k, nt_k)
            f = self.scale_convs[k](f)
            f = F.interpolate(f, size=(self.Nx, self.Nt), mode="bilinear", align_corners=False)
            fields.append(f)
        return torch.cat(fields, dim=1)


class CrossScaleFusion(nn.Module):
    def __init__(self, in_ch: int, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, width, kernel_size=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8 if width >= 8 else 1, width)
        self.se_fc1 = nn.Linear(width, max(width // 4, 4))
        self.se_fc2 = nn.Linear(max(width // 4, 4), width)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.norm(self.conv2(x)))
        se = x.mean(dim=(2, 3))
        se = F.gelu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        return x * se.unsqueeze(-1).unsqueeze(-1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes_x, modes_t):
        super().__init__()
        self.modes_x, self.modes_t = modes_x, modes_t
        self.weight = nn.Parameter(torch.randn(in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat) / (in_ch * out_ch))

    def forward(self, x):
        B, C, nx, nt = x.shape
        x_ft = rfft2(x, s=(nx, nt), norm="forward")
        mx = min(self.modes_x, x_ft.shape[2])
        mt = min(self.modes_t, x_ft.shape[3])

        out_ft = torch.zeros(B, self.weight.shape[1], nx, nt // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :mx, :mt] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :mx, :mt],
            self.weight[:, :, :mx, :mt]
        )
        return torch.fft.irfft2(out_ft, s=(nx, nt), norm="forward")


class MIFNOBlock(nn.Module):
    def __init__(self, ch, cond_ch, modes_x, modes_t):
        super().__init__()
        self.spec = SpectralConv2d(ch, ch, modes_x, modes_t)
        self.lin = nn.Conv2d(ch, ch, 1)
        self.norm = nn.GroupNorm(8 if ch >= 8 else 1, ch)
        self.cond_proj = nn.Conv2d(cond_ch, ch, 1)

    def forward(self, x, cond):
        return F.gelu(self.norm(self.spec(x) + self.lin(x) + self.cond_proj(cond)))


class MIFNO(nn.Module):
    def __init__(self, Nx, Nt, param_dim, n_scales, scale_width, width, depth, modes_x, modes_t):
        super().__init__()
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
            nn.Conv2d(width, width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 2, 1),
        )

    def forward(self, x_vec):
        fields = [enc(x_vec[:, i:i+1]) for i, enc in enumerate(self.input_encoders)]
        x = torch.cat(fields, dim=1)
        cond = self.fusion(x)
        out = cond
        for blk in self.blocks:
            out = blk(out, cond)
        return self.head(out)


# ============================================================
# MIONet model
# ============================================================
def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int):
    layers = []
    d0 = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d0, hidden_dim), nn.GELU()]
        d0 = hidden_dim
    layers += [nn.Linear(d0, out_dim)]
    return nn.Sequential(*layers)


class MIONet(nn.Module):
    def __init__(self, param_dim, latent_dim, branch_width, branch_depth, trunk_width, trunk_depth, Nx, Nt):
        super().__init__()
        self.Nx = Nx
        self.Nt = Nt
        self.latent_dim = latent_dim

        self.branch = make_mlp(param_dim, branch_width, 2 * latent_dim, branch_depth)
        self.trunk = make_mlp(2, trunk_width, 2 * latent_dim, trunk_depth)
        self.bias_head = nn.Parameter(torch.zeros(2))

    def forward(self, params, coords):
        B = params.shape[0]
        P = coords.shape[0]
        branch_out = self.branch(params).view(B, 2, self.latent_dim)
        trunk_out = self.trunk(coords).view(P, 2, self.latent_dim)
        out = torch.einsum("bcl,pcl->bcp", branch_out, trunk_out) + self.bias_head.view(1, 2, 1)
        return out.view(B, 2, self.Nx, self.Nt)


# ============================================================
# Load FEM dataset
# ============================================================
data_dir = os.path.join("results", args.split)

param_files = sorted(
    [f for f in os.listdir(data_dir) if f.startswith("params_case_")],
    key=extract_case_idx
)
temp_files = sorted(
    [f for f in os.listdir(data_dir) if f.startswith("temperature_all_case")],
    key=extract_case_idx
)
stress_files = sorted(
    [f for f in os.listdir(data_dir) if f.startswith("stress_all_case")],
    key=extract_case_idx
)

X = []
Y_temp = []
Y_stress = []

# infer Nt/Nx from latest configs later? we will use MIFNO config as base and check MIONet same
for p, t, s in zip(param_files, temp_files, stress_files):
    X.append(np.loadtxt(os.path.join(data_dir, p), delimiter=",", skiprows=1))
    Y_temp.append(np.loadtxt(os.path.join(data_dir, t)))
    Y_stress.append(np.loadtxt(os.path.join(data_dir, s)))

X = np.array(X).astype(np.float32)


# ============================================================
# Load MIFNO config / scalers / model
# ============================================================
mifno_dir = "MIFNO"
mifno_cfg_path = os.path.join(mifno_dir, "latest_config.json")
mifno_weights = os.path.join(mifno_dir, "latest_model_best.pt")

if not os.path.exists(mifno_cfg_path):
    raise FileNotFoundError("Missing MIFNO/latest_config.json")
if not os.path.exists(mifno_weights):
    raise FileNotFoundError("Missing MIFNO/latest_model_best.pt")

with open(mifno_cfg_path) as f:
    mifno_cfg = json.load(f)

Nx = mifno_cfg["Nx"]
Nt = mifno_cfg["Nt"]
param_dim = mifno_cfg["param_dim"]

Y_temp = np.array(Y_temp).reshape(len(X), Nt, Nx).transpose(0, 2, 1)
Y_stress = np.array(Y_stress).reshape(len(X), Nt, Nx).transpose(0, 2, 1)

x_mean_mifno = np.load(os.path.join(mifno_dir, "x_scaler_mean_mifno_mz.npy"))
x_std_mifno = np.load(os.path.join(mifno_dir, "x_scaler_scale_mifno_mz.npy"))
y_temp_mean_mifno = np.load(os.path.join(mifno_dir, "y_temp_scaler_mean_mz.npy"))
y_temp_std_mifno = np.load(os.path.join(mifno_dir, "y_temp_scaler_scale_mz.npy"))
y_stress_mean_mifno = np.load(os.path.join(mifno_dir, "y_stress_scaler_mean_mz.npy"))
y_stress_std_mifno = np.load(os.path.join(mifno_dir, "y_stress_scaler_scale_mz.npy"))

mifno_model = MIFNO(
    Nx=Nx, Nt=Nt, param_dim=param_dim,
    n_scales=mifno_cfg["n_scales"],
    scale_width=mifno_cfg["scale_width"],
    width=mifno_cfg["width"],
    depth=mifno_cfg["depth"],
    modes_x=mifno_cfg["modes_x"],
    modes_t=mifno_cfg["modes_t"],
).to(device)

mifno_model.load_state_dict(torch.load(mifno_weights, map_location=device), strict=True)
mifno_model.eval()
print("✅ Loaded MIFNO")


# ============================================================
# Load MIONet config / scalers / model
# ============================================================
mionet_dir = "MIONET"
mionet_cfg_path = os.path.join(mionet_dir, "latest_config_mionet.json")
mionet_weights = os.path.join(mionet_dir, "latest_model_best_mionet.pt")

if not os.path.exists(mionet_cfg_path):
    raise FileNotFoundError("Missing MIONET/latest_config_mionet.json")
if not os.path.exists(mionet_weights):
    raise FileNotFoundError("Missing MIONET/latest_model_best_mionet.pt")

with open(mionet_cfg_path) as f:
    mionet_cfg = json.load(f)

if mionet_cfg["Nx"] != Nx or mionet_cfg["Nt"] != Nt or mionet_cfg["param_dim"] != param_dim:
    raise ValueError("MIFNO and MIONet configs are inconsistent.")

x_mean_mionet = np.load(os.path.join(mionet_dir, "x_scaler_mean_mionet_mz.npy"))
x_std_mionet = np.load(os.path.join(mionet_dir, "x_scaler_scale_mionet_mz.npy"))
y_temp_mean_mionet = np.load(os.path.join(mionet_dir, "y_temp_scaler_mean_mionet_mz.npy"))
y_temp_std_mionet = np.load(os.path.join(mionet_dir, "y_temp_scaler_scale_mionet_mz.npy"))
y_stress_mean_mionet = np.load(os.path.join(mionet_dir, "y_stress_scaler_mean_mionet_mz.npy"))
y_stress_std_mionet = np.load(os.path.join(mionet_dir, "y_stress_scaler_scale_mionet_mz.npy"))

mionet_model = MIONet(
    param_dim=param_dim,
    latent_dim=mionet_cfg["latent_dim"],
    branch_width=mionet_cfg["branch_width"],
    branch_depth=mionet_cfg["branch_depth"],
    trunk_width=mionet_cfg["trunk_width"],
    trunk_depth=mionet_cfg["trunk_depth"],
    Nx=Nx, Nt=Nt,
).to(device)

mionet_model.load_state_dict(torch.load(mionet_weights, map_location=device), strict=True)
mionet_model.eval()
print("✅ Loaded MIONet")


# ============================================================
# Inference
# ============================================================
N = len(X)
print(f"✅ Loaded dataset from {data_dir}")
print(f"   N = {N}")

X_scaled_mifno = ((X - x_mean_mifno) / x_std_mifno).astype(np.float32)
X_scaled_mionet = ((X - x_mean_mionet) / x_std_mionet).astype(np.float32)

x_coords = np.linspace(0.0, 1.0, Nx, dtype=np.float32)
t_coords = np.linspace(0.0, 1.0, Nt, dtype=np.float32)
XX, TT = np.meshgrid(x_coords, t_coords, indexing="ij")
coords = np.stack([XX.ravel(), TT.ravel()], axis=1).astype(np.float32)
coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

with torch.no_grad():
    _ = mifno_model(torch.tensor(X_scaled_mifno[:1], dtype=torch.float32, device=device))
    _ = mionet_model(torch.tensor(X_scaled_mionet[:1], dtype=torch.float32, device=device), coords_t)

sync(device)
t0 = time.time()
with torch.no_grad():
    pred_mifno = mifno_model(torch.tensor(X_scaled_mifno, dtype=torch.float32, device=device)).cpu().numpy()
sync(device)
mifno_time = time.time() - t0

sync(device)
t1 = time.time()
with torch.no_grad():
    pred_mionet = mionet_model(torch.tensor(X_scaled_mionet, dtype=torch.float32, device=device), coords_t).cpu().numpy()
sync(device)
mionet_time = time.time() - t1

Y_pred_temp_mifno = pred_mifno[:, 0] * y_temp_std_mifno + y_temp_mean_mifno
Y_pred_stress_mifno = pred_mifno[:, 1] * y_stress_std_mifno + y_stress_mean_mifno

Y_pred_temp_mionet = pred_mionet[:, 0] * y_temp_std_mionet + y_temp_mean_mionet
Y_pred_stress_mionet = pred_mionet[:, 1] * y_stress_std_mionet + y_stress_mean_mionet


# ============================================================
# Metrics
# ============================================================
temp_l2_mifno = rel_l2(Y_pred_temp_mifno, Y_temp)
stress_l2_mifno = rel_l2(Y_pred_stress_mifno, Y_stress)

temp_l2_mionet = rel_l2(Y_pred_temp_mionet, Y_temp)
stress_l2_mionet = rel_l2(Y_pred_stress_mionet, Y_stress)

print("\n📊 RESULTS")
print(f"MIFNO  | L2 Temp = {temp_l2_mifno:.6f} ({temp_l2_mifno*100:.3f}%) | L2 Stress = {stress_l2_mifno:.6f} ({stress_l2_mifno*100:.3f}%)")
print(f"MIONet | L2 Temp = {temp_l2_mionet:.6f} ({temp_l2_mionet*100:.3f}%) | L2 Stress = {stress_l2_mionet:.6f} ({stress_l2_mionet*100:.3f}%)")

print("\n⏱️ Inference Performance")
print(f"MIFNO  total = {mifno_time:.4f} s | per sample = {mifno_time / N:.6f} s")
print(f"MIONet total = {mionet_time:.4f} s | per sample = {mionet_time / N:.6f} s")


# ============================================================
# Comparison plots
# ============================================================
os.makedirs("plots_case_checks", exist_ok=True)

case_id = min(max(0, args.case_id), N - 1)
t_idx = min(max(0, args.t_idx), Nt - 1)

dt = args.dt
velocity_m_per_s = args.velocity
t_axis = np.arange(Nt) * dt
x_axis = velocity_m_per_s * t_axis

surface_idx = 0
mid_idx = Nx // 2

# 1) surface temp vs time
out1 = f"plots_case_checks/compare_surface_temp_vs_time_case{case_id}.png"
plt.figure(figsize=(10, 5))
plt.plot(t_axis, Y_temp[case_id, surface_idx, :], "--", label="FEM")
plt.plot(t_axis, Y_pred_temp_mifno[case_id, surface_idx, :], label="MIFNO")
plt.plot(t_axis, Y_pred_temp_mionet[case_id, surface_idx, :], label="MIONet")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title("Surface temperature vs time", pad=18)
plt.legend()
plt.grid(True)
add_zone_lines_time(plt.gca(), t_axis)
plt.tight_layout()
plt.savefig(out1, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out1}")

# 2) surface temp vs distance
out2 = f"plots_case_checks/compare_surface_temp_vs_distance_case{case_id}.png"
plt.figure(figsize=(10, 5))
plt.plot(x_axis, Y_temp[case_id, surface_idx, :], "--", label="FEM")
plt.plot(x_axis, Y_pred_temp_mifno[case_id, surface_idx, :], label="MIFNO")
plt.plot(x_axis, Y_pred_temp_mionet[case_id, surface_idx, :], label="MIONet")
plt.xlabel("Lehr distance x (m)")
plt.ylabel("Temperature (K)")
plt.title("Surface temperature vs Lehr distance", pad=18)
plt.legend()
plt.grid(True)
add_zone_lines_distance(plt.gca(), x_axis, velocity_m_per_s)
plt.tight_layout()
plt.savefig(out2, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out2}")

# 3) stress vs time
out3 = f"plots_case_checks/compare_stress_vs_time_case{case_id}.png"
plt.figure(figsize=(10, 5))
plt.plot(t_axis, Y_stress[case_id, surface_idx, :], "--", label="FEM surface")
plt.plot(t_axis, Y_pred_stress_mifno[case_id, surface_idx, :], label="MIFNO surface")
plt.plot(t_axis, Y_pred_stress_mionet[case_id, surface_idx, :], label="MIONet surface")
plt.plot(t_axis, Y_stress[case_id, mid_idx, :], "--", label="FEM mid")
plt.plot(t_axis, Y_pred_stress_mifno[case_id, mid_idx, :], label="MIFNO mid")
plt.plot(t_axis, Y_pred_stress_mionet[case_id, mid_idx, :], label="MIONet mid")
plt.xlabel("Time (s)")
plt.ylabel("Stress (Pa)")
plt.title("Stress vs time", pad=18)
plt.legend(ncol=2)
plt.grid(True)
add_zone_lines_time(plt.gca(), t_axis)
plt.tight_layout()
plt.savefig(out3, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out3}")

# 4) one-time thickness temperature
out4 = f"plots_case_checks/compare_temp_thickness_case{case_id}_t{t_idx}.png"
plt.figure(figsize=(9, 5))
plt.plot(Y_temp[case_id, :, t_idx], "--", label="FEM")
plt.plot(Y_pred_temp_mifno[case_id, :, t_idx], label="MIFNO")
plt.plot(Y_pred_temp_mionet[case_id, :, t_idx], label="MIONet")
plt.xlabel("Thickness index")
plt.ylabel("Temperature (K)")
plt.title(f"Temperature through thickness at t_idx={t_idx}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out4, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out4}")

# 5) one-time thickness stress
out5 = f"plots_case_checks/compare_stress_thickness_case{case_id}_t{t_idx}.png"
plt.figure(figsize=(9, 5))
plt.plot(Y_stress[case_id, :, t_idx], "--", label="FEM")
plt.plot(Y_pred_stress_mifno[case_id, :, t_idx], label="MIFNO")
plt.plot(Y_pred_stress_mionet[case_id, :, t_idx], label="MIONet")
plt.xlabel("Thickness index")
plt.ylabel("Stress (Pa)")
plt.title(f"Stress through thickness at t_idx={t_idx}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out5, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out5}")

# 6) field maps comparison
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

im0 = axes[0, 0].imshow(Y_temp[case_id], aspect="auto", origin="lower")
axes[0, 0].set_title("FEM Temp")
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(Y_pred_temp_mifno[case_id], aspect="auto", origin="lower")
axes[0, 1].set_title("MIFNO Temp")
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(Y_pred_temp_mionet[case_id], aspect="auto", origin="lower")
axes[0, 2].set_title("MIONet Temp")
plt.colorbar(im2, ax=axes[0, 2])

im3 = axes[1, 0].imshow(Y_stress[case_id], aspect="auto", origin="lower")
axes[1, 0].set_title("FEM Stress")
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(Y_pred_stress_mifno[case_id], aspect="auto", origin="lower")
axes[1, 1].set_title("MIFNO Stress")
plt.colorbar(im4, ax=axes[1, 1])

im5 = axes[1, 2].imshow(Y_pred_stress_mionet[case_id], aspect="auto", origin="lower")
axes[1, 2].set_title("MIONet Stress")
plt.colorbar(im5, ax=axes[1, 2])

for ax in axes.ravel():
    ax.set_xlabel("time-index")
    ax.set_ylabel("thickness-index")

plt.tight_layout()
out6 = f"plots_case_checks/compare_field_maps_case{case_id}.png"
plt.savefig(out6, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out6}")

# 7) error maps
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

im0 = axes[0, 0].imshow(np.abs(Y_pred_temp_mifno[case_id] - Y_temp[case_id]), aspect="auto", origin="lower")
axes[0, 0].set_title("MIFNO |Temp Error|")
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(np.abs(Y_pred_temp_mionet[case_id] - Y_temp[case_id]), aspect="auto", origin="lower")
axes[0, 1].set_title("MIONet |Temp Error|")
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[1, 0].imshow(np.abs(Y_pred_stress_mifno[case_id] - Y_stress[case_id]), aspect="auto", origin="lower")
axes[1, 0].set_title("MIFNO |Stress Error|")
plt.colorbar(im2, ax=axes[1, 0])

im3 = axes[1, 1].imshow(np.abs(Y_pred_stress_mionet[case_id] - Y_stress[case_id]), aspect="auto", origin="lower")
axes[1, 1].set_title("MIONet |Stress Error|")
plt.colorbar(im3, ax=axes[1, 1])

for ax in axes.ravel():
    ax.set_xlabel("time-index")
    ax.set_ylabel("thickness-index")

plt.tight_layout()
out7 = f"plots_case_checks/compare_error_maps_case{case_id}.png"
plt.savefig(out7, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out7}")


# ============================================================
# Combined summary block charts
# ============================================================
fem_time_per_sample = 30.5
mifno_time_per_sample = mifno_time / N
mionet_time_per_sample = mionet_time / N

models = ["FEM", "MIFNO", "MIONet"]

temp_errors_pct = [0.0, temp_l2_mifno * 100.0, temp_l2_mionet * 100.0]
stress_errors_pct = [0.0, stress_l2_mifno * 100.0, stress_l2_mionet * 100.0]
times_per_sample = [fem_time_per_sample, mifno_time_per_sample, mionet_time_per_sample]
speedups = [1.0,
            fem_time_per_sample / mifno_time_per_sample,
            fem_time_per_sample / mionet_time_per_sample]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Temp accuracy
bars = axes[0, 0].bar(models, temp_errors_pct)
axes[0, 0].set_title("Temperature accuracy")
axes[0, 0].set_ylabel("Relative L2 Error (%)")
axes[0, 0].grid(True, axis="y", linestyle="--", alpha=0.5)
for b, val in zip(bars, temp_errors_pct):
    axes[0, 0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02 * max(temp_errors_pct + [1]),
                    f"{val:.3f}", ha="center", va="bottom")

# Stress accuracy
bars = axes[0, 1].bar(models, stress_errors_pct)
axes[0, 1].set_title("Stress accuracy")
axes[0, 1].set_ylabel("Relative L2 Error (%)")
axes[0, 1].grid(True, axis="y", linestyle="--", alpha=0.5)
for b, val in zip(bars, stress_errors_pct):
    axes[0, 1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02 * max(stress_errors_pct + [1]),
                    f"{val:.3f}", ha="center", va="bottom")

# Time
bars = axes[1, 0].bar(models, times_per_sample)
axes[1, 0].set_title("Computation time")
axes[1, 0].set_ylabel("Time per sample (s)")
axes[1, 0].grid(True, axis="y", linestyle="--", alpha=0.5)
for b, val in zip(bars, times_per_sample):
    axes[1, 0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02 * max(times_per_sample + [1e-3]),
                    f"{val:.4f}", ha="center", va="bottom")

# Speed-up
bars = axes[1, 1].bar(models, speedups)
axes[1, 1].set_title("Speed-up vs FEM")
axes[1, 1].set_ylabel("Speed-up (x)")
axes[1, 1].grid(True, axis="y", linestyle="--", alpha=0.5)
for b, val in zip(bars, speedups):
    axes[1, 1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02 * max(speedups + [1]),
                    f"{val:.2f}x", ha="center", va="bottom")

plt.tight_layout()
out_summary = "plots_case_checks/compare_efficiency_summary.png"
plt.savefig(out_summary, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out_summary}")