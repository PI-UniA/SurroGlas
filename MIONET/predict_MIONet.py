# MIONET/predict_MIONet_operator.py
# ============================================================
# Predict/evaluate TRUE MIONet (4 independent branch nets,
# multiplicative fusion) trained by:
#   MIONET/train_eval_MIONET_operator.py
#
# Examples:
#   python MIONET/predict_MIONet_operator.py --device mps --split test_unseen
#   python MIONET/predict_MIONet_operator.py --device mps --split train --case_id 3 --t_idx 190
#   python MIONET/predict_MIONet_operator.py --device mps --params "120,0.6,3.67e-8,10"
#   python MIONET/predict_MIONet_operator.py --device mps --weights best
# ============================================================

import os, time, json, argparse, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ─────────────────────────────────────────
# Device + sync
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


# ─────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────
def extract_case_idx(fname: str) -> int:
    m = re.search(r"case[_]?(\d+)", fname)
    return int(m.group(1)) if m else 10**9

def load_split(results_dir: str, Nx: int, Nt: int):
    param_files  = sorted([f for f in os.listdir(results_dir) if f.startswith("params_case_")],        key=extract_case_idx)
    temp_files   = sorted([f for f in os.listdir(results_dir) if f.startswith("temperature_all_case")], key=extract_case_idx)
    stress_files = sorted([f for f in os.listdir(results_dir) if f.startswith("stress_all_case")],      key=extract_case_idx)

    if not (len(param_files) == len(temp_files) == len(stress_files) > 0):
        raise RuntimeError(
            f"Dataset mismatch/empty in {results_dir}: "
            f"params={len(param_files)}, temp={len(temp_files)}, stress={len(stress_files)}"
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

    X  = np.stack(X_params).astype(np.float32)
    Yt = np.stack(Y_temperatures).astype(np.float32).reshape(len(X), Nt, Nx).transpose(0, 2, 1)
    Ys = np.stack(Y_stresses).astype(np.float32).reshape(len(X), Nt, Nx).transpose(0, 2, 1)
    return X, Yt, Ys

def rel_l2(pred, true):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)

def make_xt_grid(Nx: int, Nt: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, Nx, dtype=np.float32)
    t = np.linspace(0.0, 1.0, Nt, dtype=np.float32)
    Xg, Tg = np.meshgrid(x, t, indexing="ij")
    return np.stack([Xg, Tg], axis=-1).reshape(-1, 2)   # (Nx*Nt, 2)

def make_mlp(in_dim: int, width: int, depth: int, out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.GELU()]
        d = width
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════
#  TRUE MIONet ARCHITECTURE — identical to train_eval_MIONET_operator.py
#
#  branches[c][i] : R¹ → Rᵖ   (one branch per input per channel)
#  fusion:  B^c(u) = b₁^c(u₁) ⊙ b₂^c(u₂) ⊙ b₃^c(u₃) ⊙ b₄^c(u₄)
#  output:  u^c(y) = B^c(u) · τ(y) + bias^c
# ═══════════════════════════════════════════════════════════════

class MIONet(nn.Module):
    def __init__(self,
                 input_dim:    int = 4,
                 p:            int = 128,
                 branch_width: int = 128,
                 branch_depth: int = 3,
                 trunk_width:  int = 256,
                 trunk_depth:  int = 4,
                 n_outputs:    int = 2):
        super().__init__()
        self.p         = p
        self.input_dim = input_dim
        self.n_outputs = n_outputs

        # shared trunk: R² → Rᵖ
        self.trunk = make_mlp(2, trunk_width, trunk_depth, p)

        # independent branch nets: one per input per output channel
        self.branches = nn.ModuleList([
            nn.ModuleList([
                make_mlp(1, branch_width, branch_depth, p)
                for _ in range(input_dim)
            ])
            for _ in range(n_outputs)
        ])

        self.bias = nn.Parameter(torch.zeros(n_outputs))

    def forward(self,
                x_params: torch.Tensor,   # (B, input_dim)
                xt:       torch.Tensor,   # (P, 2)
                Nx:       int,
                Nt:       int) -> torch.Tensor:
        B = x_params.shape[0]

        # trunk encoding
        phi = self.trunk(xt)   # (P, p)

        channel_outputs = []
        for c in range(self.n_outputs):
            # multiplicative fusion across all branch nets for channel c
            fused = torch.ones(B, self.p, device=x_params.device)
            for i, branch in enumerate(self.branches[c]):
                fused = fused * branch(x_params[:, i:i+1])   # ⊙

            # dot product with trunk + bias
            out_flat = fused @ phi.T + self.bias[c]          # (B, P)
            channel_outputs.append(out_flat.view(B, Nx, Nt))

        return torch.stack(channel_outputs, dim=1)           # (B, n_outputs, Nx, Nt)


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device",     type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
parser.add_argument("--split",      type=str, default="test_unseen", choices=["train", "test_unseen"])
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--case_id",    type=int, default=1)
parser.add_argument("--t_idx",      type=int, default=190)
parser.add_argument("--weights",    type=str, default="best", choices=["best", "last"])
parser.add_argument("--params",     type=str, default="",
                    help='Optional custom params "htc,epsilon,sigma,alpha" to predict ONE sample (no FEM truth needed)')
args = parser.parse_args()

device = pick_device(args.device)
if device.type == "cpu":
    torch.set_num_threads(8)


# ─────────────────────────────────────────
# Load config + scalers
# ─────────────────────────────────────────
save_dir = "MIONET"
cfg_path = os.path.join(save_dir, "config.json")
if not os.path.exists(cfg_path):
    raise FileNotFoundError(f"Missing {cfg_path}. Run train_eval_MIONET_operator.py first.")

with open(cfg_path) as f:
    CFG = json.load(f)

Nx, Nt = int(CFG["Nx"]), int(CFG["Nt"])
p      = int(CFG["p"])

x_mean      = np.load(os.path.join(save_dir, "x_scaler_mean.npy"))
x_std       = np.load(os.path.join(save_dir, "x_scaler_scale.npy"))
temp_mean   = np.load(os.path.join(save_dir, "y_temp_mean.npy")).reshape(Nx, Nt)
temp_std    = np.load(os.path.join(save_dir, "y_temp_scale.npy")).reshape(Nx, Nt)
stress_mean = np.load(os.path.join(save_dir, "y_stress_mean.npy")).reshape(Nx, Nt)
stress_std  = np.load(os.path.join(save_dir, "y_stress_scale.npy")).reshape(Nx, Nt)

w_best = os.path.join(save_dir, f"mionet_operator_dual_{device.type}_best.pt")
w_last = os.path.join(save_dir, f"mionet_operator_dual_{device.type}.pt")
weights_fp = w_best if (args.weights == "best" and os.path.exists(w_best)) else w_last
if not os.path.exists(weights_fp):
    raise FileNotFoundError(f"No weights found at {weights_fp}")

print(f"🚀 TRUE MIONet inference | device={device} | weights={weights_fp} | split={args.split}")


# ─────────────────────────────────────────
# Build model + load weights
# ─────────────────────────────────────────
model = MIONet(
    input_dim    = int(CFG["input_dim"]),
    p            = p,
    branch_width = int(CFG["branch_width"]),
    branch_depth = int(CFG["branch_depth"]),
    trunk_width  = int(CFG["trunk_width"]),
    trunk_depth  = int(CFG["trunk_depth"]),
    n_outputs    = 2,
).to(device)

state = torch.load(weights_fp, map_location=device)
model.load_state_dict(state, strict=True)
model.eval()
print(f"✅ Loaded weights from {weights_fp}")

XT = torch.tensor(make_xt_grid(Nx, Nt), dtype=torch.float32, device=device)  # (Nx*Nt, 2)


# ─────────────────────────────────────────
# Single custom prediction (no FEM truth)
# ─────────────────────────────────────────
if args.params.strip():
    parts = [v.strip() for v in args.params.split(",")]
    if len(parts) != 4:
        raise ValueError('Expected --params "htc,epsilon,sigma,alpha" (4 comma-separated values)')

    new_input  = np.array([[float(v) for v in parts]], dtype=np.float32)
    new_scaled = (new_input - x_mean) / x_std

    with torch.inference_mode():
        xb = torch.tensor(new_scaled, dtype=torch.float32, device=device)
        sync(device)
        t0  = time.time()
        out = model(xb, XT, Nx, Nt)
        sync(device)
        infer_time = time.time() - t0

    pred_s      = out.detach().cpu().numpy()[0]          # (2, Nx, Nt) scaled
    pred_temp   = pred_s[0] * temp_std   + temp_mean
    pred_stress = pred_s[1] * stress_std + stress_mean

    print(f"⏱️  Inference time (1 sample): {infer_time:.6f}s")
    print(f"📌 Params: htc={new_input[0,0]}, eps={new_input[0,1]}, "
          f"sigma={new_input[0,2]}, alpha={new_input[0,3]}")

    t_idx = int(args.t_idx)
    plt.figure(figsize=(8, 5))
    plt.plot(pred_temp[:, t_idx], label="Pred Temp")
    plt.title(f"MIONet Temp prediction at t_idx={t_idx}")
    plt.xlabel("x-index"); plt.ylabel("Temperature (K)")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(pred_stress[:, t_idx], label="Pred Stress")
    plt.title(f"MIONet Stress prediction at t_idx={t_idx}")
    plt.xlabel("x-index"); plt.ylabel("Stress (Pa)")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    raise SystemExit(0)


# ─────────────────────────────────────────
# Load dataset split + batch inference
# ─────────────────────────────────────────
split_dir = os.path.join("results", args.split)
X, Yt_true, Ys_true = load_split(split_dir, Nx, Nt)
X_scaled = (X - x_mean) / x_std

bs          = max(1, int(args.batch_size))
pred_chunks = []

# warmup
with torch.inference_mode():
    xb0 = torch.tensor(X_scaled[:min(bs, len(X_scaled))], dtype=torch.float32, device=device)
    _ = model(xb0, XT, Nx, Nt)
    sync(device)

sync(device)
t0 = time.time()
with torch.inference_mode():
    for i in range(0, len(X_scaled), bs):
        xb  = torch.tensor(X_scaled[i:i+bs], dtype=torch.float32, device=device)
        out = model(xb, XT, Nx, Nt)
        pred_chunks.append(out.detach().cpu())
sync(device)
inference_time = time.time() - t0

Y_pred_s    = torch.cat(pred_chunks, dim=0).numpy()      # (N, 2, Nx, Nt) scaled
Yt_pred     = Y_pred_s[:, 0] * temp_std   + temp_mean
Ys_pred     = Y_pred_s[:, 1] * stress_std + stress_mean

print("\n🔎 Range check:")
print("TEMP   true min/max:", Yt_true.min(), Yt_true.max())
print("TEMP   pred min/max:", Yt_pred.min(), Yt_pred.max())
print("STRESS true min/max:", Ys_true.min(), Ys_true.max())
print("STRESS pred min/max:", Ys_pred.min(), Ys_pred.max())


# ─────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────
temp_l2   = rel_l2(Yt_pred,  Yt_true)
stress_l2 = rel_l2(Ys_pred,  Ys_true)

print("\n📊 ====== MIONet Evaluation ======")
print(f"Split: {args.split} | device={device.type} | weights={args.weights}")
print(f"L2(T): {temp_l2:.6e} | L2(σ): {stress_l2:.6e}  "
      f"({temp_l2*100:.3f}%, {stress_l2*100:.3f}%)")
print(f"⏱️  Total inference: {inference_time:.3f}s for {len(X_scaled)} samples "
      f"({inference_time/max(len(X_scaled),1):.6f}s/sample)")

metrics = {
    "model":                       "MIONet",
    "split":                       args.split,
    "device":                      device.type,
    "weights":                     args.weights,
    "num_samples":                 int(len(X_scaled)),
    "batch_size":                  int(bs),
    "inference_time_total_s":      float(inference_time),
    "inference_time_per_sample_s": float(inference_time / max(len(X_scaled), 1)),
    "l2_temp":                     float(temp_l2),
    "l2_stress":                   float(stress_l2),
    "l2_temp_percent":             float(temp_l2   * 100),
    "l2_stress_percent":           float(stress_l2 * 100),
    "p":                           int(p),
    "n_branch_nets":               int(CFG["input_dim"]) * 2,
}
out_json = os.path.join(save_dir, f"metrics_predict_{args.split}_{device.type}_{args.weights}.json")
with open(out_json, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"✅ Saved metrics → {out_json}")


# ─────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────
case_id = min(int(args.case_id), len(X_scaled) - 1)
t_idx   = int(args.t_idx)

print(f"\n📌 Plotting case_id={case_id} | t_idx={t_idx} | split={args.split}")
print("params [htc, epsilon, sigma, alpha]:", X[case_id].reshape(-1))

plt.figure(figsize=(8, 5))
plt.plot(Yt_true[case_id, :, t_idx], "--", label="True Temp")
plt.plot(Yt_pred[case_id, :, t_idx],       label="Pred Temp")
plt.title(f"MIONet Temperature at t_idx={t_idx} ({args.split}, case={case_id})")
plt.xlabel("x-index"); plt.ylabel("Temperature (K)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
plt.plot(Ys_true[case_id, :, t_idx], "--", label="True Stress")
plt.plot(Ys_pred[case_id, :, t_idx],       label="Pred Stress")
plt.title(f"MIONet Stress at t_idx={t_idx} ({args.split}, case={case_id})")
plt.xlabel("x-index"); plt.ylabel("Stress (Pa)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

temp_slice_l2   = rel_l2(Yt_pred[case_id, :, t_idx],  Yt_true[case_id, :, t_idx])
stress_slice_l2 = rel_l2(Ys_pred[case_id, :, t_idx],  Ys_true[case_id, :, t_idx])
print(f"📈 Relative L2 at one time-slice (Temp):   {temp_slice_l2:.6f} ({temp_slice_l2*100:.4f}%)")
print(f"📈 Relative L2 at one time-slice (Stress): {stress_slice_l2:.6f} ({stress_slice_l2*100:.4f}%)")
