import os
import json
import time
import argparse
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
parser.add_argument("--device",      type=str, default="auto")
parser.add_argument("--split",       type=str, default="train")
parser.add_argument("--case_id",     type=int, default=0)
parser.add_argument("--t_idx",       type=int, default=3299)
parser.add_argument("--save_dir",    type=str, default="MIFNO")
parser.add_argument("--infer_batch", type=int, default=4,
                    help="Samples per GPU batch. Reduce to 2 or 1 if OOM.")
args = parser.parse_args()

device = pick_device(args.device)
print(f"🚀 Inference | device={device}")


# ============================================================
# LOAD CONFIG
# ============================================================
save_dir     = args.save_dir
cfg_path     = os.path.join(save_dir, "latest_config.json")
weights_path = os.path.join(save_dir, "latest_model_best.pt")

if not os.path.exists(cfg_path):
    raise FileNotFoundError(f"❌ latest_config.json not found in {save_dir}")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"❌ latest_model_best.pt not found in {save_dir}")

with open(cfg_path) as f:
    cfg = json.load(f)

Nx        = cfg["Nx"]
Nt        = cfg["Nt"]
param_dim = cfg["param_dim"]

print(f"✅ Loaded config → {cfg_path}")
print(f"   Nx={Nx}, Nt={Nt}, param_dim={param_dim}")
print(f"✅ Using weights → {weights_path}")


# ============================================================
# LOAD SCALERS
# ============================================================
x_mean = np.load(os.path.join(save_dir, "x_scaler_mean_mifno_mz.npy"))
x_std  = np.load(os.path.join(save_dir, "x_scaler_scale_mifno_mz.npy"))

T_mean = float(np.load(os.path.join(save_dir, "y_temp_mean_mz.npy")).flat[0])
T_std  = float(np.load(os.path.join(save_dir, "y_temp_std_mz.npy")).flat[0])
S_mean = float(np.load(os.path.join(save_dir, "y_stress_mean_mz.npy")).flat[0])
S_std  = float(np.load(os.path.join(save_dir, "y_stress_std_mz.npy")).flat[0])

print("✅ Loaded scalers")
print(f"   T: mean={T_mean:.4f}, std={T_std:.4f}")
print(f"   S: mean={S_mean:.4f}, std={S_std:.4f}")


# ============================================================
# MODEL DEFINITION — must exactly match train_MIFNO.py
# ============================================================

class BatchedMultiscaleEncoder(nn.Module):
    BASE_NX = 8
    BASE_NT = 16

    def __init__(self, Nx, Nt, param_dim, n_scales, scale_width, embed_dim=128):
        super().__init__()
        self.Nx          = Nx
        self.Nt          = Nt
        self.param_dim   = param_dim
        self.n_scales    = n_scales
        self.scale_width = scale_width

        self.mlp = nn.Sequential(
            nn.Linear(param_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        base_flat = self.BASE_NX * self.BASE_NT
        self.scale_heads = nn.ModuleList([
            nn.Linear(embed_dim, scale_width * base_flat)
            for _ in range(n_scales)
        ])

        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(scale_width, scale_width, kernel_size=3, padding=1),
                nn.GELU(),
            )
            for _ in range(n_scales)
        ])

    def forward(self, x_vec):
        B   = x_vec.shape[0]
        emb = self.mlp(x_vec)

        scale_fields = []
        for k in range(self.n_scales):
            field_k = self.scale_heads[k](emb)
            field_k = field_k.view(B, self.scale_width, self.BASE_NX, self.BASE_NT)
            field_k = F.interpolate(
                field_k, size=(self.Nx, self.Nt),
                mode="bilinear", align_corners=False
            )
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
        self.modes_x = modes_x
        self.modes_t = modes_t
        scale = 1.0 / (in_ch * out_ch) ** 0.5
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, nx, nt = x.shape
        x_ft = rfft2(x, s=(nx, nt), norm="ortho")

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
        h = self.drop(F.gelu(h))
        return h


class MIFNO(nn.Module):
    def __init__(self, Nx, Nt, param_dim, n_scales, scale_width,
                 width, depth, modes_x, modes_t, dropout=0.1):
        super().__init__()
        self.encoder = BatchedMultiscaleEncoder(
            Nx, Nt, param_dim, n_scales, scale_width, embed_dim=128
        )

        in_ch = n_scales * scale_width
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

    def forward(self, x_vec):
        multi_input = self.encoder(x_vec)
        cond = self.fusion(multi_input)
        x    = cond
        for blk in self.blocks:
            x = blk(x, cond)
        return self.head(x)


# ============================================================
# BUILD + LOAD MODEL
# ============================================================
model = MIFNO(
    Nx          = Nx,
    Nt          = Nt,
    param_dim   = param_dim,
    n_scales    = cfg["n_scales"],
    scale_width = cfg["scale_width"],
    width       = cfg["width"],
    depth       = cfg["depth"],
    modes_x     = cfg["modes_x"],
    modes_t     = cfg["modes_t"],
    dropout     = cfg.get("dropout", 0.1),
).to(device)

state_dict = torch.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict, strict=True)
model.eval()
print(f"✅ Model loaded | params: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# LOAD DATA
# ============================================================
data_dir = os.path.join("results", args.split)

param_files  = sorted([f for f in os.listdir(data_dir) if "params"      in f])
temp_files   = sorted([f for f in os.listdir(data_dir) if "temperature" in f])
stress_files = sorted([f for f in os.listdir(data_dir) if "stress"      in f])

X        = []
Y_temp   = []
Y_stress = []

for p, t, s in zip(param_files, temp_files, stress_files):
    X.append(np.loadtxt(os.path.join(data_dir, p), delimiter=",", skiprows=1))
    Y_temp.append(np.loadtxt(os.path.join(data_dir, t)))
    Y_stress.append(np.loadtxt(os.path.join(data_dir, s)))

X        = np.array(X).astype(np.float32)
Y_temp   = np.array(Y_temp).reshape(len(X), Nt, Nx).transpose(0, 2, 1)   # (N, Nx, Nt)
Y_stress = np.array(Y_stress).reshape(len(X), Nt, Nx).transpose(0, 2, 1) # (N, Nx, Nt)

print(f"✅ Loaded {len(X)} samples from {data_dir}")


# ============================================================
# BATCHED INFERENCE  (avoids OOM on large Nx*Nt fields)
# ============================================================
X_scaled = (X - x_mean) / x_std
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)  # CPU; moved to GPU batch-by-batch

# GPU warm-up
with torch.no_grad():
    _ = model(X_tensor[:1].to(device))
if device.type == "cuda":
    torch.cuda.synchronize()

print(f"   Batched inference | infer_batch={args.infer_batch} "
      f"(pass --infer_batch 1 if still OOM)")

t0 = time.time()
pred_chunks = []
with torch.no_grad():
    for i in range(0, len(X_tensor), args.infer_batch):
        xb = X_tensor[i : i + args.infer_batch].to(device)
        pred_chunks.append(model(xb).cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()   # free activations after each batch

if device.type == "cuda":
    torch.cuda.synchronize()
inference_time = time.time() - t0

pred = torch.cat(pred_chunks, dim=0).numpy()   # (N, 2, Nx, Nt)

# Denormalize with global scalars
Y_pred_temp   = pred[:, 0] * T_std + T_mean   # (N, Nx, Nt)
Y_pred_stress = pred[:, 1] * S_std + S_mean   # (N, Nx, Nt)

print(f"\n⏱️  Inference Performance")
print(f"   Total time      = {inference_time:.4f} s")
print(f"   Samples         = {len(X)}")
print(f"   Time per sample = {inference_time / len(X):.6f} s")


# ============================================================
# METRICS
# ============================================================
def rel_l2(pred, true):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)

temp_l2   = rel_l2(Y_pred_temp,   Y_temp)
stress_l2 = rel_l2(Y_pred_stress, Y_stress)

print(f"\n📊 Results")
print(f"   Relative L2 Temperature = {temp_l2:.6f}  ({temp_l2*100:.3f}%)")
print(f"   Relative L2 Stress      = {stress_l2:.6f}  ({stress_l2*100:.3f}%)")


# ============================================================
# PLOTS
# ============================================================
os.makedirs("plots_case_checks", exist_ok=True)

case_id = min(max(0, args.case_id), len(X) - 1)
t_idx   = min(max(0, args.t_idx),   Nt - 1)

dt               = 0.1
velocity_m_per_s = 0.24
t_axis           = np.arange(Nt) * dt
x_axis           = velocity_m_per_s * t_axis

surface_idx = 0
mid_idx     = Nx // 2


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"   ✅ {path}")


print("\n📈 Saving plots ...")

# 1) Through-thickness temperature at one time step
plt.figure(figsize=(8, 5))
plt.plot(Y_temp[case_id, :, t_idx],      "--", label="FEM")
plt.plot(Y_pred_temp[case_id, :, t_idx],       label="MIFNO")
plt.xlabel("Thickness index"); plt.ylabel("Temperature (K)")
plt.title(f"Temperature through thickness | case={case_id}, t_idx={t_idx}")
plt.legend(); plt.grid(True)
savefig(f"plots_case_checks/temp_thickness_case{case_id}_t{t_idx}.png")

# 2) Through-thickness stress at one time step
plt.figure(figsize=(8, 5))
plt.plot(Y_stress[case_id, :, t_idx],      "--", label="FEM")
plt.plot(Y_pred_stress[case_id, :, t_idx],       label="MIFNO")
plt.xlabel("Thickness index"); plt.ylabel("Stress (Pa)")
plt.title(f"Stress through thickness | case={case_id}, t_idx={t_idx}")
plt.legend(); plt.grid(True)
savefig(f"plots_case_checks/stress_thickness_case{case_id}_t{t_idx}.png")

# 3) Surface temperature vs time
plt.figure(figsize=(9, 5))
plt.plot(t_axis, Y_temp[case_id, surface_idx, :],      "--", label="FEM surface")
plt.plot(t_axis, Y_pred_temp[case_id, surface_idx, :],       label="MIFNO surface")
plt.xlabel("Time (s)"); plt.ylabel("Temperature (K)")
plt.title(f"Surface temperature vs time | case={case_id}")
plt.legend(); plt.grid(True)
savefig(f"plots_case_checks/surface_temp_time_case{case_id}.png")

# 4) Surface temperature vs Lehr distance
plt.figure(figsize=(9, 5))
plt.plot(x_axis, Y_temp[case_id, surface_idx, :],      "--", label="FEM surface")
plt.plot(x_axis, Y_pred_temp[case_id, surface_idx, :],       label="MIFNO surface")
plt.xlabel("Lehr distance x (m)"); plt.ylabel("Temperature (K)")
plt.title(f"Surface temperature vs Lehr distance | case={case_id}")
plt.legend(); plt.grid(True)
savefig(f"plots_case_checks/surface_temp_distance_case{case_id}.png")

# 5) Stress vs time at surface and mid-plane
plt.figure(figsize=(9, 5))
plt.plot(t_axis, Y_stress[case_id, surface_idx, :],      "--", label="FEM surface")
plt.plot(t_axis, Y_pred_stress[case_id, surface_idx, :],       label="MIFNO surface")
plt.plot(t_axis, Y_stress[case_id, mid_idx, :],           "--", label="FEM mid-plane")
plt.plot(t_axis, Y_pred_stress[case_id, mid_idx, :],            label="MIFNO mid-plane")
plt.xlabel("Time (s)"); plt.ylabel("Stress (Pa)")
plt.title(f"Stress vs time | case={case_id}")
plt.legend(); plt.grid(True)
savefig(f"plots_case_checks/stress_time_case{case_id}.png")

# 6) FEM temperature field map
plt.figure(figsize=(10, 4))
plt.imshow(Y_temp[case_id], aspect="auto", origin="lower", cmap="RdYlBu_r")
plt.colorbar(label="Temperature (K)")
plt.xlabel("Time index"); plt.ylabel("Thickness index")
plt.title(f"FEM temperature field | case={case_id}")
savefig(f"plots_case_checks/fem_temp_field_case{case_id}.png")

# 7) MIFNO temperature field map
plt.figure(figsize=(10, 4))
plt.imshow(Y_pred_temp[case_id], aspect="auto", origin="lower", cmap="RdYlBu_r")
plt.colorbar(label="Temperature (K)")
plt.xlabel("Time index"); plt.ylabel("Thickness index")
plt.title(f"MIFNO temperature field | case={case_id}")
savefig(f"plots_case_checks/mifno_temp_field_case{case_id}.png")

# 8) FEM stress field map
plt.figure(figsize=(10, 4))
plt.imshow(Y_stress[case_id], aspect="auto", origin="lower", cmap="PuOr")
plt.colorbar(label="Stress (Pa)")
plt.xlabel("Time index"); plt.ylabel("Thickness index")
plt.title(f"FEM stress field | case={case_id}")
savefig(f"plots_case_checks/fem_stress_field_case{case_id}.png")

# 9) MIFNO stress field map
plt.figure(figsize=(10, 4))
plt.imshow(Y_pred_stress[case_id], aspect="auto", origin="lower", cmap="PuOr")
plt.colorbar(label="Stress (Pa)")
plt.xlabel("Time index"); plt.ylabel("Thickness index")
plt.title(f"MIFNO stress field | case={case_id}")
savefig(f"plots_case_checks/mifno_stress_field_case{case_id}.png")

# 10) Temperature absolute error map
plt.figure(figsize=(10, 4))
plt.imshow(np.abs(Y_pred_temp[case_id] - Y_temp[case_id]), aspect="auto", origin="lower")
plt.colorbar(label="|Error| (K)")
plt.xlabel("Time index"); plt.ylabel("Thickness index")
plt.title(f"Temperature absolute error | case={case_id}")
savefig(f"plots_case_checks/temp_error_map_case{case_id}.png")

# 11) Stress absolute error map
plt.figure(figsize=(10, 4))
plt.imshow(np.abs(Y_pred_stress[case_id] - Y_stress[case_id]), aspect="auto", origin="lower")
plt.colorbar(label="|Error| (Pa)")
plt.xlabel("Time index"); plt.ylabel("Thickness index")
plt.title(f"Stress absolute error | case={case_id}")
savefig(f"plots_case_checks/stress_error_map_case{case_id}.png")


print("\n✅ All done.")