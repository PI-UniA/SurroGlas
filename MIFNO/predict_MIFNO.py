import os, time, json, argparse, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2, irfft2
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CLI
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
        raise RuntimeError("MPS requested but not available.")
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

parser = argparse.ArgumentParser()
parser.add_argument("--device",     type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
parser.add_argument("--split",      type=str, default="train", choices=["train", "test_unseen"])
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--case_id",    type=int, default=1)
parser.add_argument("--t_idx",      type=int, default=190)
args = parser.parse_args()

device = pick_device(args.device)
if device.type == "cpu":
    torch.set_num_threads(8)

# ─────────────────────────────────────────
# Config — must match train_MIFNO.py exactly
# ─────────────────────────────────────────
save_dir  = "MIFNO"
cfg_path  = os.path.join(save_dir, "mifno_config.json")

if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    Nx          = cfg["Nx"]
    Nt          = cfg["Nt"]
    param_dim   = cfg["param_dim"]
    n_scales    = cfg["n_scales"]
    scale_width = cfg["scale_width"]
    width       = cfg["width"]
    depth       = cfg["depth"]
    modes_x     = cfg["modes_x"]
    modes_t     = cfg["modes_t"]
    print(f"✅ Loaded config from {cfg_path}")
else:
    # fallback defaults (mirror train_MIFNO.py hyperparameters)
    print(f"⚠️  Config not found at {cfg_path}, using defaults.")
    Nx, Nt        = 49, 500
    param_dim     = 4
    n_scales      = 3
    scale_width   = 16
    width         = 64
    depth         = 6
    modes_x       = 16
    modes_t       = 16

weights_fp = os.path.join(save_dir, f"mifno_model_dual_{device.type}_best.pt")
if not os.path.exists(weights_fp):
    alt = os.path.join(save_dir, f"mifno_model_dual_{device.type}.pt")
    if os.path.exists(alt):
        weights_fp = alt
        print(f"⚠️  Best weights not found, falling back to last: {alt}")
    else:
        raise FileNotFoundError(f"No weights found at {weights_fp} or {alt}")

test_dir = os.path.join("results", args.split)

print(f"🚀 Inference | device={device} | weights={weights_fp} | split={args.split}")

# ─────────────────────────────────────────
# Load training scalers
# ─────────────────────────────────────────
x_mean        = np.load(os.path.join(save_dir, "x_scaler_mean_mifno.npy"))
x_std         = np.load(os.path.join(save_dir, "x_scaler_scale_mifno.npy"))
y_temp_mean   = np.load(os.path.join(save_dir, "y_temp_scaler_mean.npy")).reshape(Nx, Nt)
y_temp_std    = np.load(os.path.join(save_dir, "y_temp_scaler_scale.npy")).reshape(Nx, Nt)
y_stress_mean = np.load(os.path.join(save_dir, "y_stress_scaler_mean.npy")).reshape(Nx, Nt)
y_stress_std  = np.load(os.path.join(save_dir, "y_stress_scaler_scale.npy")).reshape(Nx, Nt)


# ═══════════════════════════════════════════════════════════════
#  MIFNO ARCHITECTURE  — identical to train_MIFNO.py
# ═══════════════════════════════════════════════════════════════

class MultiscaleInputEncoder(nn.Module):
    def __init__(self, Nx: int, Nt: int, n_scales: int, scale_width: int):
        super().__init__()
        self.Nx, self.Nt = Nx, Nt
        self.n_scales    = n_scales
        self.scale_width = scale_width

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
        B = x_scalar.shape[0]
        scale_fields = []
        for k in range(self.n_scales):
            nx_k = max(self.Nx // (2 ** k), 4)
            nt_k = max(self.Nt // (2 ** k), 4)
            field_k = self.scale_lifters[k](x_scalar)
            field_k = field_k.view(B, 1, nx_k, nt_k)
            field_k = self.scale_convs[k](field_k)
            field_k = F.interpolate(field_k, size=(self.Nx, self.Nt),
                                    mode="bilinear", align_corners=False)
            scale_fields.append(field_k)
        return torch.cat(scale_fields, dim=1)


class CrossScaleFusion(nn.Module):
    def __init__(self, in_ch: int, width: int, Nx: int, Nt: int):
        super().__init__()
        self.conv1  = nn.Conv2d(in_ch, width, kernel_size=1)
        self.conv2  = nn.Conv2d(width,  width, kernel_size=3, padding=1)
        self.norm   = nn.GroupNorm(8, width)
        self.se_fc1 = nn.Linear(width, width // 4)
        self.se_fc2 = nn.Linear(width // 4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = F.gelu(self.conv1(x))
        x  = F.gelu(self.norm(self.conv2(x)))
        se = x.mean(dim=(2, 3))
        se = F.gelu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        return x * se.unsqueeze(-1).unsqueeze(-1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes_x: int = 16, modes_t: int = 16):
        super().__init__()
        self.modes_x, self.modes_t = modes_x, modes_t
        self.weight = nn.Parameter(
            torch.randn(in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat)
            / (in_ch * out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, nx, nt = x.shape
        x_ft   = rfft2(x, s=(nx, nt), norm="forward")
        out_ft = torch.zeros(B, self.weight.shape[1], nx, nt // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes_x, :self.modes_t] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes_x, :self.modes_t],
            self.weight,
        )
        return irfft2(out_ft, s=(nx, nt), norm="forward")


class MIFNOBlock(nn.Module):
    def __init__(self, ch: int, cond_ch: int, modes_x: int = 16, modes_t: int = 16):
        super().__init__()
        self.spec      = SpectralConv2d(ch, ch, modes_x, modes_t)
        self.lin       = nn.Conv2d(ch, ch, 1)
        self.norm      = nn.GroupNorm(8, ch)
        self.cond_proj = nn.Conv2d(cond_ch, ch, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return F.gelu(
            self.norm(self.spec(x) + self.lin(x) + self.cond_proj(cond))
        )


class MIFNO(nn.Module):
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

        self.input_encoders = nn.ModuleList([
            MultiscaleInputEncoder(Nx, Nt, n_scales, scale_width)
            for _ in range(param_dim)
        ])

        in_ch = param_dim * n_scales * scale_width
        self.fusion = CrossScaleFusion(in_ch, width, Nx, Nt)

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


# ─────────────────────────────────────────
# Load model weights
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

state = torch.load(weights_fp, map_location=device)
model.load_state_dict(state, strict=True)
model.eval()
print(f"✅ Loaded weights from {weights_fp}")

# ─────────────────────────────────────────
# Load test data
# ─────────────────────────────────────────
def extract_case_idx(fname: str) -> int:
    m = re.search(r"case(\d+)", fname)
    return int(m.group(1)) if m else 10**9

param_files  = sorted([f for f in os.listdir(test_dir) if f.startswith("params_case_")],        key=extract_case_idx)
temp_files   = sorted([f for f in os.listdir(test_dir) if f.startswith("temperature_all_case")], key=extract_case_idx)
stress_files = sorted([f for f in os.listdir(test_dir) if f.startswith("stress_all_case")],      key=extract_case_idx)

if not (len(param_files) == len(temp_files) == len(stress_files) > 0):
    raise RuntimeError(
        f"Dataset mismatch in {test_dir}: "
        f"params={len(param_files)}, temp={len(temp_files)}, stress={len(stress_files)}"
    )

X_params, Y_temperatures, Y_stresses = [], [], []
for p, t, s in zip(param_files, temp_files, stress_files):
    params   = np.loadtxt(os.path.join(test_dir, p), skiprows=1, delimiter=",")
    temps    = np.loadtxt(os.path.join(test_dir, t))
    stresses = np.loadtxt(os.path.join(test_dir, s))
    if params.ndim == 0:
        params = np.expand_dims(params, axis=0)
    X_params.append(params)
    Y_temperatures.append(temps)
    Y_stresses.append(stresses)

N       = len(X_params)
X_te    = np.stack(X_params).astype(np.float32)
Yt      = np.stack(Y_temperatures).astype(np.float32).reshape(N, Nt, Nx).transpose(0, 2, 1)
Ys      = np.stack(Y_stresses).astype(np.float32).reshape(N, Nt, Nx).transpose(0, 2, 1)

X_te_scaled = (X_te - x_mean) / x_std

# ─────────────────────────────────────────
# Inference
# ─────────────────────────────────────────
BATCH_SIZE  = args.batch_size
pred_chunks = []

# warmup
with torch.inference_mode():
    xb0 = torch.tensor(X_te_scaled[:min(BATCH_SIZE, N)], dtype=torch.float32, device=device)
    _ = model(xb0)
    sync(device)

sync(device)
t0 = time.time()
with torch.inference_mode():
    for i in range(0, N, BATCH_SIZE):
        xb  = torch.tensor(X_te_scaled[i:i+BATCH_SIZE], dtype=torch.float32, device=device)
        out = model(xb)
        pred_chunks.append(out.detach().cpu())
sync(device)
inference_time = time.time() - t0

Y_pred_s = torch.cat(pred_chunks, dim=0).numpy()

# Inverse-transform to physical units
Y_pred_temp   = Y_pred_s[:, 0] * y_temp_std   + y_temp_mean
Y_pred_stress = Y_pred_s[:, 1] * y_stress_std + y_stress_mean
Y_true_temp   = Yt
Y_true_stress = Ys

print("\n🔎 Range check:")
print("TEMP   true min/max:", Y_true_temp.min(),   Y_true_temp.max())
print("TEMP   pred min/max:", Y_pred_temp.min(),   Y_pred_temp.max())
print("STRESS true min/max:", Y_true_stress.min(), Y_true_stress.max())
print("STRESS pred min/max:", Y_pred_stress.min(), Y_pred_stress.max())

# ─────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────
def rel_l2(pred, true):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)

temp_l2   = rel_l2(Y_pred_temp,   Y_true_temp)
stress_l2 = rel_l2(Y_pred_stress, Y_true_stress)

print("\n📊 ====== MIFNO Evaluation ======")
print(f"Split: {args.split} | device={device.type}")
print(f"L2(T): {temp_l2:.6e} | L2(σ): {stress_l2:.6e}  "
      f"({temp_l2*100:.3f}%, {stress_l2*100:.3f}%)")
print(f"⏱️  Total inference: {inference_time:.3f}s for {N} samples "
      f"({inference_time/max(N,1):.6f}s/sample)")

metrics = {
    "split":                      args.split,
    "device":                     device.type,
    "num_samples":                int(N),
    "batch_size":                 int(BATCH_SIZE),
    "inference_time_total_s":     float(inference_time),
    "inference_time_per_sample_s":float(inference_time / max(N, 1)),
    "l2_temp":                    float(temp_l2),
    "l2_stress":                  float(stress_l2),
    "l2_temp_percent":            float(temp_l2   * 100),
    "l2_stress_percent":          float(stress_l2 * 100),
}
out_json = f"metrics_results_MIFNO_{args.split}_{device.type}.json"
with open(out_json, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"✅ Saved metrics → {out_json}")

# ─────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────
case_id = min(args.case_id, N - 1)
t_idx   = int(args.t_idx)

print("\n📌 Plotting this case:")
print("split:", args.split)
print("case_id:", case_id)
print("params [htc, epsilon, sigma, alpha]:", X_te[case_id].reshape(-1))

plt.figure(figsize=(8, 5))
plt.plot(Y_true_temp[case_id, :, t_idx],   "--", label="True Temp")
plt.plot(Y_pred_temp[case_id, :, t_idx],         label="Pred Temp")
plt.title(f"Temperature at t_idx={t_idx} ({args.split})")
plt.xlabel("x-index"); plt.ylabel("Temperature (K)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
plt.plot(Y_true_stress[case_id, :, t_idx], "--", label="True Stress")
plt.plot(Y_pred_stress[case_id, :, t_idx],       label="Pred Stress")
plt.title(f"Stress at t_idx={t_idx} ({args.split})")
plt.xlabel("x-index"); plt.ylabel("Stress (Pa)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

temp_slice_l2   = rel_l2(Y_pred_temp[case_id, :, t_idx],   Y_true_temp[case_id, :, t_idx])
stress_slice_l2 = rel_l2(Y_pred_stress[case_id, :, t_idx], Y_true_stress[case_id, :, t_idx])
print(f"📈 Relative L2 at one time-slice (Temp):   {temp_slice_l2:.6f} ({temp_slice_l2*100:.4f}%)")
print(f"📈 Relative L2 at one time-slice (Stress): {stress_slice_l2:.6f} ({stress_slice_l2*100:.4f}%)")
