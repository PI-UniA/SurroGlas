import os
import json
import time
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2, irfft2
import matplotlib.pyplot as plt


# ============================================================
# DEVICE
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


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--split", type=str, default="test_unseen")
parser.add_argument("--case_id", type=int, default=0)
parser.add_argument("--t_idx", type=int, default=190)
args = parser.parse_args()

device = pick_device(args.device)

print(f"🚀 Inference | device={device}")


# ============================================================
# LOAD LATEST FILES
# ============================================================
save_dir = "MIFNO"

cfg_path = os.path.join(save_dir, "latest_config.json")
weights_path = os.path.join(save_dir, "latest_model_best.pt")

if not os.path.exists(cfg_path):
    raise FileNotFoundError("❌ latest_config.json not found")

if not os.path.exists(weights_path):
    raise FileNotFoundError("❌ latest_model_best.pt not found")

with open(cfg_path) as f:
    cfg = json.load(f)

Nx = cfg["Nx"]
Nt = cfg["Nt"]
param_dim = cfg["param_dim"]

print(f"✅ Loaded config → {cfg_path}")
print(f"   Nx={Nx}, Nt={Nt}, param_dim={param_dim}")

print(f"✅ Using weights → {weights_path}")


# ============================================================
# LOAD SCALERS
# ============================================================
x_mean = np.load(os.path.join(save_dir, "x_scaler_mean_mifno_mz.npy"))
x_std = np.load(os.path.join(save_dir, "x_scaler_scale_mifno_mz.npy"))

y_temp_mean = np.load(os.path.join(save_dir, "y_temp_scaler_mean_mz.npy"))
y_temp_std = np.load(os.path.join(save_dir, "y_temp_scaler_scale_mz.npy"))

y_stress_mean = np.load(os.path.join(save_dir, "y_stress_scaler_mean_mz.npy"))
y_stress_std = np.load(os.path.join(save_dir, "y_stress_scaler_scale_mz.npy"))

print("✅ Loaded scalers")


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

            f = F.interpolate(f, size=(self.Nx, self.Nt), mode="bilinear")
            fields.append(f)

        return torch.cat(fields, dim=1)


class CrossScaleFusion(nn.Module):
    def __init__(self, in_ch: int, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, width, kernel_size=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, width)

        # ✅ ADD THIS (missing part)
        self.se_fc1 = nn.Linear(width, width // 4)
        self.se_fc2 = nn.Linear(width // 4, width)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.norm(self.conv2(x)))

        # ✅ ADD THIS (missing part)
        se = x.mean(dim=(2, 3))
        se = F.gelu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))

        return x * se.unsqueeze(-1).unsqueeze(-1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes_x, modes_t):
        super().__init__()
        self.modes_x, self.modes_t = modes_x, modes_t
        self.weight = nn.Parameter(
            torch.randn(in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, nx, nt = x.shape
        x_ft = rfft2(x)

        out_ft = torch.zeros(B, self.weight.shape[1], nx, nt//2 + 1,
                             device=x.device, dtype=torch.cfloat)

        out_ft[:, :, :self.modes_x, :self.modes_t] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes_x, :self.modes_t],
            self.weight
        )

        return irfft2(out_ft, s=(nx, nt))


class MIFNOBlock(nn.Module):
    def __init__(self, ch, cond_ch, modes_x, modes_t):
        super().__init__()
        self.spec = SpectralConv2d(ch, ch, modes_x, modes_t)
        self.lin = nn.Conv2d(ch, ch, 1)
        self.norm = nn.GroupNorm(8, ch)
        self.cond_proj = nn.Conv2d(cond_ch, ch, 1)

    def forward(self, x, cond):
        return F.gelu(self.norm(self.spec(x) + self.lin(x) + self.cond_proj(cond)))


class MIFNO(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_encoders = nn.ModuleList([
            MultiscaleInputEncoder(Nx, Nt, cfg["n_scales"], cfg["scale_width"])
            for _ in range(param_dim)
        ])

        in_ch = param_dim * cfg["n_scales"] * cfg["scale_width"]
        width = cfg["width"]

        self.fusion = CrossScaleFusion(in_ch, width)

        self.blocks = nn.ModuleList([
            MIFNOBlock(width, width, cfg["modes_x"], cfg["modes_t"])
            for _ in range(cfg["depth"])
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
# BUILD + LOAD MODEL
# ============================================================
# build model here
model = MIFNO().to(device)
state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.eval()
print("✅ Model loaded")
# ============================================================
# LOAD DATA
# ============================================================
data_dir = os.path.join("results", args.split)

param_files = sorted([f for f in os.listdir(data_dir) if "params" in f])
temp_files = sorted([f for f in os.listdir(data_dir) if "temperature" in f])
stress_files = sorted([f for f in os.listdir(data_dir) if "stress" in f])

X = []
Y_temp = []
Y_stress = []

for p, t, s in zip(param_files, temp_files, stress_files):
    X.append(np.loadtxt(os.path.join(data_dir, p), delimiter=",", skiprows=1))
    Y_temp.append(np.loadtxt(os.path.join(data_dir, t)))
    Y_stress.append(np.loadtxt(os.path.join(data_dir, s)))

X = np.array(X).astype(np.float32)
Y_temp = np.array(Y_temp).reshape(len(X), Nt, Nx).transpose(0, 2, 1)
Y_stress = np.array(Y_stress).reshape(len(X), Nt, Nx).transpose(0, 2, 1)

print(f"✅ Loaded dataset from {data_dir}")
print(f"   N = {len(X)}")


# ============================================================
# INFERENCE + TIMING
# ============================================================
X_scaled = (X - x_mean) / x_std

# Warm-up (important for GPU timing)
with torch.no_grad():
    _ = model(torch.tensor(X_scaled[:1], dtype=torch.float32, device=device))
    
if device.type == "cuda":
    torch.cuda.synchronize()

start_time = time.time()

with torch.no_grad():
    pred = model(torch.tensor(X_scaled, dtype=torch.float32, device=device)).cpu().numpy()

if device.type == "cuda":
    torch.cuda.synchronize()

inference_time = time.time() - start_time

# Post-processing
Y_pred_temp = pred[:, 0] * y_temp_std + y_temp_mean
Y_pred_stress = pred[:, 1] * y_stress_std + y_stress_mean

print("\n⏱️ Inference Performance")
print(f"Total time      = {inference_time:.4f} s")
print(f"Samples         = {len(X)}")
print(f"Time per sample = {inference_time / len(X):.6f} s")

# ============================================================
# METRICS
# ============================================================
def rel_l2(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)

temp_l2 = rel_l2(Y_pred_temp, Y_temp)
stress_l2 = rel_l2(Y_pred_stress, Y_stress)

print("\n📊 RESULTS")
print(f"L2 Temp   = {temp_l2:.6f} ({temp_l2*100:.3f}%)")
print(f"L2 Stress = {stress_l2:.6f} ({stress_l2*100:.3f}%)")

# ============================================================
# PLOTS
# ============================================================
os.makedirs("plots_case_checks", exist_ok=True)

case_id = min(max(0, args.case_id), len(X) - 1)
t_idx = min(max(0, args.t_idx), Nt - 1)

dt = 0.1
velocity_m_per_s = 0.24

t_axis = np.arange(Nt) * dt
x_axis = velocity_m_per_s * t_axis

surface_idx = 0
mid_idx = Nx // 2
last_idx = Nx - 1

# ------------------------------------------------------------
# 1) Through-thickness profile at one time
# ------------------------------------------------------------
temp_path = f"plots_case_checks/mifno_temp_case{case_id}_t{t_idx}.png"

plt.figure(figsize=(8, 5))
plt.plot(Y_temp[case_id, :, t_idx], "--", label="FEM")
plt.plot(Y_pred_temp[case_id, :, t_idx], label="MIFNO")
plt.xlabel("Thickness index")
plt.ylabel("Temperature (K)")
plt.title(f"Temperature through thickness at t_idx={t_idx}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(temp_path, dpi=300)
plt.close()
print(f"✅ Saved: {temp_path}")

stress_path = f"plots_case_checks/mifno_stress_case{case_id}_t{t_idx}.png"

plt.figure(figsize=(8, 5))
plt.plot(Y_stress[case_id, :, t_idx], "--", label="FEM")
plt.plot(Y_pred_stress[case_id, :, t_idx], label="MIFNO")
plt.xlabel("Thickness index")
plt.ylabel("Stress (Pa)")
plt.title(f"Stress through thickness at t_idx={t_idx}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(stress_path, dpi=300)
plt.close()
print(f"✅ Saved: {stress_path}")

# ------------------------------------------------------------
# 2) Surface temperature vs time
# ------------------------------------------------------------
surf_temp_time_path = f"plots_case_checks/mifno_surface_temp_vs_time_case{case_id}.png"

plt.figure(figsize=(9, 5))
plt.plot(t_axis, Y_temp[case_id, surface_idx, :], "--", label="FEM surface")
plt.plot(t_axis, Y_pred_temp[case_id, surface_idx, :], label="MIFNO surface")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title("Surface temperature vs time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(surf_temp_time_path, dpi=300)
plt.close()
print(f"✅ Saved: {surf_temp_time_path}")

# ------------------------------------------------------------
# 3) Surface temperature vs Lehr distance
# ------------------------------------------------------------
surf_temp_dist_path = f"plots_case_checks/mifno_surface_temp_vs_distance_case{case_id}.png"

plt.figure(figsize=(9, 5))
plt.plot(x_axis, Y_temp[case_id, surface_idx, :], "--", label="FEM surface")
plt.plot(x_axis, Y_pred_temp[case_id, surface_idx, :], label="MIFNO surface")
plt.xlabel("Lehr distance x (m)")
plt.ylabel("Temperature (K)")
plt.title("Surface temperature vs Lehr distance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(surf_temp_dist_path, dpi=300)
plt.close()
print(f"✅ Saved: {surf_temp_dist_path}")

# ------------------------------------------------------------
# 4) Stress vs time at surface and mid-plane
# ------------------------------------------------------------
stress_time_path = f"plots_case_checks/mifno_stress_vs_time_case{case_id}.png"

plt.figure(figsize=(9, 5))
plt.plot(t_axis, Y_stress[case_id, surface_idx, :], "--", label="FEM surface")
plt.plot(t_axis, Y_pred_stress[case_id, surface_idx, :], label="MIFNO surface")
plt.plot(t_axis, Y_stress[case_id, mid_idx, :], "--", label="FEM mid-plane")
plt.plot(t_axis, Y_pred_stress[case_id, mid_idx, :], label="MIFNO mid-plane")
plt.xlabel("Time (s)")
plt.ylabel("Stress (Pa)")
plt.title("Stress vs time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(stress_time_path, dpi=300)
plt.close()
print(f"✅ Saved: {stress_time_path}")

# ------------------------------------------------------------
# 5) FEM temperature field map
# ------------------------------------------------------------
fem_temp_map_path = f"plots_case_checks/fem_temp_field_case{case_id}.png"

plt.figure(figsize=(10, 4))
plt.imshow(Y_temp[case_id], aspect="auto", origin="lower")
plt.colorbar(label="Temperature (K)")
plt.xlabel("Time index")
plt.ylabel("Thickness index")
plt.title("FEM temperature field")
plt.tight_layout()
plt.savefig(fem_temp_map_path, dpi=300)
plt.close()
print(f"✅ Saved: {fem_temp_map_path}")

# ------------------------------------------------------------
# 6) MIFNO temperature field map
# ------------------------------------------------------------
mifno_temp_map_path = f"plots_case_checks/mifno_temp_field_case{case_id}.png"

plt.figure(figsize=(10, 4))
plt.imshow(Y_pred_temp[case_id], aspect="auto", origin="lower")
plt.colorbar(label="Temperature (K)")
plt.xlabel("Time index")
plt.ylabel("Thickness index")
plt.title("MIFNO temperature field")
plt.tight_layout()
plt.savefig(mifno_temp_map_path, dpi=300)
plt.close()
print(f"✅ Saved: {mifno_temp_map_path}")

# ------------------------------------------------------------
# 7) FEM stress field map
# ------------------------------------------------------------
fem_stress_map_path = f"plots_case_checks/fem_stress_field_case{case_id}.png"

plt.figure(figsize=(10, 4))
plt.imshow(Y_stress[case_id], aspect="auto", origin="lower")
plt.colorbar(label="Stress (Pa)")
plt.xlabel("Time index")
plt.ylabel("Thickness index")
plt.title("FEM stress field")
plt.tight_layout()
plt.savefig(fem_stress_map_path, dpi=300)
plt.close()
print(f"✅ Saved: {fem_stress_map_path}")

# ------------------------------------------------------------
# 8) MIFNO stress field map
# ------------------------------------------------------------
mifno_stress_map_path = f"plots_case_checks/mifno_stress_field_case{case_id}.png"

plt.figure(figsize=(10, 4))
plt.imshow(Y_pred_stress[case_id], aspect="auto", origin="lower")
plt.colorbar(label="Stress (Pa)")
plt.xlabel("Time index")
plt.ylabel("Thickness index")
plt.title("MIFNO stress field")
plt.tight_layout()
plt.savefig(mifno_stress_map_path, dpi=300)
plt.close()
print(f"✅ Saved: {mifno_stress_map_path}")