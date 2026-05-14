import os
import json
import time
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--case_id", type=int, default=0)
parser.add_argument("--t_idx", type=int, default=3299)
args = parser.parse_args()

device = pick_device(args.device)
print(f"🚀 Inference | device={device}")


# ============================================================
# Load latest files
# ============================================================
save_dir = "MIONET"
cfg_path = os.path.join(save_dir, "latest_config_mionet.json")
weights_path = os.path.join(save_dir, "latest_model_best_mionet.pt")

if not os.path.exists(cfg_path):
    raise FileNotFoundError("❌ latest_config_mionet.json not found")
if not os.path.exists(weights_path):
    raise FileNotFoundError("❌ latest_model_best_mionet.pt not found")

with open(cfg_path) as f:
    cfg = json.load(f)

Nx = cfg["Nx"]
Nt = cfg["Nt"]
param_dim = cfg["param_dim"]

print(f"✅ Loaded config → {cfg_path}")
print(f"   Nx={Nx}, Nt={Nt}, param_dim={param_dim}")
print(f"✅ Using weights → {weights_path}")


# ============================================================
# Load scalers
# ============================================================
x_mean = np.load(os.path.join(save_dir, "x_scaler_mean_mionet_mz.npy"))
x_std = np.load(os.path.join(save_dir, "x_scaler_scale_mionet_mz.npy"))

y_temp_mean = np.load(os.path.join(save_dir, "y_temp_scaler_mean_mionet_mz.npy"))
y_temp_std = np.load(os.path.join(save_dir, "y_temp_scaler_scale_mionet_mz.npy"))

y_stress_mean = np.load(os.path.join(save_dir, "y_stress_scaler_mean_mionet_mz.npy"))
y_stress_std = np.load(os.path.join(save_dir, "y_stress_scaler_scale_mionet_mz.npy"))

print("✅ Loaded scalers")


# ============================================================
# Model
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
        out = out.view(B, 2, self.Nx, self.Nt)
        return out


model = MIONet(
    param_dim=param_dim,
    latent_dim=cfg["latent_dim"],
    branch_width=cfg["branch_width"],
    branch_depth=cfg["branch_depth"],
    trunk_width=cfg["trunk_width"],
    trunk_depth=cfg["trunk_depth"],
    Nx=Nx,
    Nt=Nt,
).to(device)

state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.eval()
print("✅ Model loaded")


# ============================================================
# Load data
# ============================================================
def extract_case_idx(fname: str) -> int:
    m = re.search(r"case[_]?(\d+)|case(\d+)", fname)
    if m:
        for g in m.groups():
            if g is not None:
                return int(g)
    return 10**9


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
# Inference + timing
# ============================================================
X_scaled = ((X - x_mean) / x_std).astype(np.float32)

x_coords = np.linspace(0.0, 1.0, Nx, dtype=np.float32)
t_coords = np.linspace(0.0, 1.0, Nt, dtype=np.float32)
XX, TT = np.meshgrid(x_coords, t_coords, indexing="ij")
coords = np.stack([XX.ravel(), TT.ravel()], axis=1).astype(np.float32)

coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

with torch.no_grad():
    _ = model(
        torch.tensor(X_scaled[:1], dtype=torch.float32, device=device),
        coords_t
    )

if device.type == "cuda":
    torch.cuda.synchronize()

start_time = time.time()

with torch.no_grad():
    pred = model(
        torch.tensor(X_scaled, dtype=torch.float32, device=device),
        coords_t
    ).cpu().numpy()

if device.type == "cuda":
    torch.cuda.synchronize()

inference_time = time.time() - start_time

Y_pred_temp = pred[:, 0] * y_temp_std + y_temp_mean
Y_pred_stress = pred[:, 1] * y_stress_std + y_stress_mean

print("\n⏱️ Inference Performance")
print(f"Total time      = {inference_time:.4f} s")
print(f"Samples         = {len(X)}")
print(f"Time per sample = {inference_time / len(X):.6f} s")


# ============================================================
# Metrics
# ============================================================
def rel_l2(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)

temp_l2 = rel_l2(Y_pred_temp, Y_temp)
stress_l2 = rel_l2(Y_pred_stress, Y_stress)

print("\n📊 RESULTS")
print(f"L2 Temp   = {temp_l2:.6f} ({temp_l2*100:.3f}%)")
print(f"L2 Stress = {stress_l2:.6f} ({stress_l2*100:.3f}%)")


# ============================================================
# Annealing-Lehr-style plots
# ============================================================
os.makedirs("plots_case_checks", exist_ok=True)

case_id = min(max(0, args.case_id), len(X) - 1)
t_idx = min(max(0, args.t_idx), Nt - 1)

# ---- physical axes ----
dt = 0.1
velocity_m_per_s = 0.16417

t_axis = np.arange(Nt) * dt
x_axis = velocity_m_per_s * t_axis

surface_idx = 0
mid_idx = Nx // 2
last_idx = Nx - 1

# Optional zone timing for labels
ZONE_TIME_WINDOWS = {
    "A1": (0.0,    54.7),
    "A2": (54.7,  109.6),
    "B1": (109.6, 182.6),
    "B2": (182.6, 255.57),
    "C1": (255.57, 328.5),
}

def add_zone_lines_time(ax):
    for zone, (t0, t1) in ZONE_TIME_WINDOWS.items():
        if t0 < t_axis[-1]:
            center = 0.5 * (max(t0, t_axis[0]) + min(t1, t_axis[-1]))
            ax.text(center, 1.02, zone,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10)
        if t_axis[0] <= t1 <= t_axis[-1]:
            ax.axvline(t1, linestyle="--", linewidth=1, color="gray")

def add_zone_lines_distance(ax):
    for zone, (t0, t1) in ZONE_TIME_WINDOWS.items():
        x0 = velocity_m_per_s * t0
        x1 = velocity_m_per_s * t1
        if x0 < x_axis[-1]:
            center = 0.5 * (max(x0, x_axis[0]) + min(x1, x_axis[-1]))
            ax.text(center, 1.02, zone,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10)
        if x_axis[0] <= x1 <= x_axis[-1]:
            ax.axvline(x1, linestyle="--", linewidth=1, color="gray")


# ------------------------------------------------------------
# 1) Surface temperature vs time
# ------------------------------------------------------------
out1 = f"plots_case_checks/mionet_surface_temp_vs_time_case{case_id}.png"

plt.figure(figsize=(9, 5))
plt.plot(t_axis, Y_temp[case_id, surface_idx, :], "--", label="FEM surface")
plt.plot(t_axis, Y_pred_temp[case_id, surface_idx, :], label="MIONet surface")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title("Annealing Lehr: Surface temperature vs time", pad=18)
plt.legend()
plt.grid(True)
add_zone_lines_time(plt.gca())
plt.tight_layout()
plt.savefig(out1, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out1}")


# ------------------------------------------------------------
# 2) Surface temperature vs Lehr distance
# ------------------------------------------------------------
out2 = f"plots_case_checks/mionet_surface_temp_vs_distance_case{case_id}.png"

plt.figure(figsize=(9, 5))
plt.plot(x_axis, Y_temp[case_id, surface_idx, :], "--", label="FEM surface")
plt.plot(x_axis, Y_pred_temp[case_id, surface_idx, :], label="MIONet surface")
plt.xlabel("Lehr distance x (m)")
plt.ylabel("Temperature (K)")
plt.title("Annealing Lehr: Surface temperature vs distance", pad=18)
plt.legend()
plt.grid(True)
add_zone_lines_distance(plt.gca())
plt.tight_layout()
plt.savefig(out2, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out2}")


# ------------------------------------------------------------
# 3) Stress vs time at surface and mid-plane
# ------------------------------------------------------------
out3 = f"plots_case_checks/mionet_stress_vs_time_case{case_id}.png"

plt.figure(figsize=(9, 5))
plt.plot(t_axis, Y_stress[case_id, surface_idx, :], "--", label="FEM surface")
plt.plot(t_axis, Y_pred_stress[case_id, surface_idx, :], label="MIONet surface")
plt.plot(t_axis, Y_stress[case_id, mid_idx, :], "--", label="FEM mid-plane")
plt.plot(t_axis, Y_pred_stress[case_id, mid_idx, :], label="MIONet mid-plane")
plt.xlabel("Time (s)")
plt.ylabel("Stress (Pa)")
plt.title("Annealing Lehr: Stress vs time", pad=18)
plt.legend()
plt.grid(True)
add_zone_lines_time(plt.gca())
plt.tight_layout()
plt.savefig(out3, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out3}")


# ------------------------------------------------------------
# 4) Stress vs Lehr distance at surface and mid-plane
# ------------------------------------------------------------
out4 = f"plots_case_checks/mionet_stress_vs_distance_case{case_id}.png"

plt.figure(figsize=(9, 5))
plt.plot(x_axis, Y_stress[case_id, surface_idx, :], "--", label="FEM surface")
plt.plot(x_axis, Y_pred_stress[case_id, surface_idx, :], label="MIONet surface")
plt.plot(x_axis, Y_stress[case_id, mid_idx, :], "--", label="FEM mid-plane")
plt.plot(x_axis, Y_pred_stress[case_id, mid_idx, :], label="MIONet mid-plane")
plt.xlabel("Lehr distance x (m)")
plt.ylabel("Stress (Pa)")
plt.title("Annealing Lehr: Stress vs distance", pad=18)
plt.legend()
plt.grid(True)
add_zone_lines_distance(plt.gca())
plt.tight_layout()
plt.savefig(out4, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out4}")


# ------------------------------------------------------------
# 5) FEM temperature field map
# ------------------------------------------------------------
out5 = f"plots_case_checks/fem_temp_field_case{case_id}.png"

plt.figure(figsize=(10, 4))
plt.imshow(
    Y_temp[case_id],
    aspect="auto",
    origin="lower", cmap="RdYlBu_r",
    extent=[t_axis[0], t_axis[-1], 0, Nx - 1]
)
plt.colorbar(label="Temperature (K)")
plt.xlabel("Time (s)")
plt.ylabel("Thickness index")
plt.title("FEM temperature field", pad=14)
plt.tight_layout()
plt.savefig(out5, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out5}")


# ------------------------------------------------------------
# 6) MIONet temperature field map
# ------------------------------------------------------------
out6 = f"plots_case_checks/mionet_temp_field_case{case_id}.png"

plt.figure(figsize=(10, 4))
plt.imshow(
    Y_pred_temp[case_id],
    aspect="auto",
    origin="lower",
    cmap="RdYlBu_r",
    extent=[t_axis[0], t_axis[-1], 0, Nx - 1]
)
plt.colorbar(label="Temperature (K)")
plt.xlabel("Time (s)")
plt.ylabel("Thickness index")
plt.title("MIONet temperature field", pad=14)
plt.tight_layout()
plt.savefig(out6, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out6}")


# ------------------------------------------------------------
# 7) FEM stress field map
# ------------------------------------------------------------
out7 = f"plots_case_checks/fem_stress_field_case{case_id}.png"

plt.figure(figsize=(10, 4))
plt.imshow(
    Y_stress[case_id],
    aspect="auto",
    origin="lower",
    cmap="PuOr",
    extent=[t_axis[0], t_axis[-1], 0, Nx - 1]
)
plt.colorbar(label="Stress (Pa)")
plt.xlabel("Time (s)")
plt.ylabel("Thickness index")
plt.title("FEM stress field", pad=14)
plt.tight_layout()
plt.savefig(out7, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out7}")


# ------------------------------------------------------------
# 8) MIONet stress field map
# ------------------------------------------------------------
out8 = f"plots_case_checks/mionet_stress_field_case{case_id}.png"

plt.figure(figsize=(10, 4))
plt.imshow(
    Y_pred_stress[case_id],
    aspect="auto",
    origin="lower",
    cmap="PuOr",
    extent=[t_axis[0], t_axis[-1], 0, Nx - 1]
)
plt.colorbar(label="Stress (Pa)")
plt.xlabel("Time (s)")
plt.ylabel("Thickness index")
plt.title("MIONet stress field", pad=14)
plt.tight_layout()
plt.savefig(out8, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out8}")