# MIONET/train_eval_MIONET_operator.py
# ============================================================
# TRUE Multi-Input Operator Network (MIONet) — Jin et al. 2022
# 4 outputs (Temp, Stress)
#
# Architecture (true MIONet):
#   - 4 independent Branch nets, one per scalar input parameter
#   - Fusion: element-wise product of all branch outputs  ⊙
#   - Trunk net: encodes coordinates (x, t)
#   - Output: dot(fused_branch, trunk) + bias  per channel
#
# Key difference from DeepONet:
#   DeepONet: single branch( [u1,u2,u3,u4] ) → p
#   MIONet:   branch1(u1) ⊙ branch2(u2) ⊙ branch3(u3) ⊙ branch4(u4) → p
#
# Trains on:    results/train
# Evaluates on: results/train and results/test_unseen
# Supports CPU / MPS / CUDA via --device
# ============================================================

import os, time, json, argparse, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0) HYPERPARAMETERS / SETTINGS
# ============================================================
CFG = {
    # data
    "Nx": 49,
    "Nt": 500,
    "input_dim": 4,           # number of scalar inputs = number of branch nets

    # operator net sizes
    "p": 128,                 # branch/trunk output dim (basis functions)
    "branch_width": 128,      # width per branch net (smaller — each sees 1 scalar)
    "branch_depth": 3,        # depth per branch net
    "trunk_width": 256,
    "trunk_depth": 4,

    # training
    "epochs": 2000,
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "batch_size": 16,
    "seed": 42,

    # evaluation/inference
    "infer_batch_size": 8,
    "save_dir": "MIONET",
    "train_dir": os.path.join("results", "train"),
    "test_dir":  os.path.join("results", "test_unseen"),

    # loss weights
    "wT": 0.5,
    "wS": 0.5,
}


# ============================================================
# Device selection + sync
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


# ============================================================
# Utilities
# ============================================================
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

    X      = np.stack(X_params).astype(np.float32)                                                    # (N, 4)
    Y_temp = np.stack(Y_temperatures).astype(np.float32).reshape(len(X), Nt, Nx).transpose(0, 2, 1)  # (N, Nx, Nt)
    Y_strs = np.stack(Y_stresses).astype(np.float32).reshape(len(X), Nt, Nx).transpose(0, 2, 1)      # (N, Nx, Nt)
    return X, Y_temp, Y_strs

def rel_l2(pred, true):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)

def make_xt_grid(Nx: int, Nt: int) -> np.ndarray:
    x  = np.linspace(0.0, 1.0, Nx,  dtype=np.float32)
    t  = np.linspace(0.0, 1.0, Nt,  dtype=np.float32)
    Xg, Tg = np.meshgrid(x, t, indexing="ij")   # (Nx, Nt)
    return np.stack([Xg, Tg], axis=-1).reshape(-1, 2)  # (Nx*Nt, 2)

def make_mlp(in_dim: int, width: int, depth: int, out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.GELU()]
        d = width
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════
#  TRUE MIONet ARCHITECTURE  — Jin et al. 2022
#
#  Each scalar input uᵢ has its OWN branch net bᵢ : R → Rᵖ
#  Fusion is the element-wise product across all branch outputs:
#
#      B(u) = b₁(u₁) ⊙ b₂(u₂) ⊙ b₃(u₃) ⊙ b₄(u₄)   ∈ Rᵖ
#
#  For 2 output fields we maintain 2 independent fused branches:
#
#      Bᶜ(u) = b₁ᶜ(u₁) ⊙ b₂ᶜ(u₂) ⊙ b₃ᶜ(u₃) ⊙ b₄ᶜ(u₄)   c ∈ {T, σ}
#
#  Final output at coordinate y = (x,t):
#
#      uᶜ(y) = Bᶜ(u) · τ(y) + biasᶜ
#
#  where τ : R² → Rᵖ is the shared trunk net.
# ═══════════════════════════════════════════════════════════════

class MIONet(nn.Module):
    """
    True Multi-Input Operator Network with 2 output fields.

    Parameters
    ----------
    input_dim    : number of scalar inputs (= number of branch nets per channel)
    p            : branch / trunk output dimension (basis size)
    branch_width : hidden width of each individual branch MLP
    branch_depth : depth of each individual branch MLP
    trunk_width  : hidden width of trunk MLP
    trunk_depth  : depth of trunk MLP
    n_outputs    : number of output fields (2: temperature + stress)
    """
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

        # ── Trunk: shared across output channels ──────────────────────
        # τ(x,t) : R² → Rᵖ
        self.trunk = make_mlp(2, trunk_width, trunk_depth, p)

        # ── Branch nets: one set per output channel ────────────────────
        # branches[c][i] encodes scalar input uᵢ for output channel c
        # Each branch: R¹ → Rᵖ  (input is a single scalar)
        self.branches = nn.ModuleList([
            nn.ModuleList([
                make_mlp(1, branch_width, branch_depth, p)
                for _ in range(input_dim)
            ])
            for _ in range(n_outputs)
        ])

        # ── Output biases: one per channel ────────────────────────────
        self.bias = nn.Parameter(torch.zeros(n_outputs))

    def forward(self,
                x_params: torch.Tensor,   # (B, input_dim)  — scalar parameters
                xt:       torch.Tensor,   # (P, 2)          — coordinate grid
                Nx:       int,
                Nt:       int) -> torch.Tensor:
        """
        Returns: (B, n_outputs, Nx, Nt)
        """
        B = x_params.shape[0]
        P = xt.shape[0]   # Nx * Nt

        # ── Trunk encoding ─────────────────────────────────────────────
        phi = self.trunk(xt)   # (P, p)

        # ── Branch encoding + multiplicative fusion per channel ────────
        channel_outputs = []
        for c in range(self.n_outputs):
            # start with ones for the multiplicative identity
            fused = torch.ones(B, self.p, device=x_params.device)   # (B, p)

            for i, branch in enumerate(self.branches[c]):
                # each branch receives one scalar: x_params[:, i:i+1] → (B,1)
                b_i = branch(x_params[:, i:i+1])   # (B, p)
                fused = fused * b_i                 # ⊙ element-wise product

            # dot product with trunk: (B,p) × (P,p) → (B,P)
            out_flat = torch.einsum("bp,pp->bp", fused, phi.T) \
                       if False else fused @ phi.T  # (B, P)
            out_flat = out_flat + self.bias[c]       # broadcast bias

            channel_outputs.append(out_flat.view(B, Nx, Nt))   # (B, Nx, Nt)

        return torch.stack(channel_outputs, dim=1)   # (B, n_outputs, Nx, Nt)


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--device",     type=str,   default="auto", choices=["auto", "cpu", "cuda", "mps"])
parser.add_argument("--threads",    type=int,   default=8)
parser.add_argument("--epochs",     type=int,   default=None)
parser.add_argument("--lr",         type=float, default=None)
parser.add_argument("--batch_size", type=int,   default=None)
args = parser.parse_args()

if args.epochs     is not None: CFG["epochs"]     = args.epochs
if args.lr         is not None: CFG["lr"]         = args.lr
if args.batch_size is not None: CFG["batch_size"] = args.batch_size

device = pick_device(args.device)
if device.type == "cpu":
    torch.set_num_threads(args.threads)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])


# ============================================================
# Paths
# ============================================================
Nx, Nt    = CFG["Nx"], CFG["Nt"]
save_dir  = CFG["save_dir"]
os.makedirs(save_dir, exist_ok=True)

weights_path = os.path.join(save_dir, f"mionet_operator_dual_{device.type}.pt")
best_path    = os.path.join(save_dir, f"mionet_operator_dual_{device.type}_best.pt")

with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump({**CFG, "device": device.type}, f, indent=4)

total_branches = CFG["input_dim"] * 2   # 2 channels × 4 inputs
print(f"🚀 TRUE MIONet train+eval | device={device} | epochs={CFG['epochs']}")
print(f"   p={CFG['p']} | {total_branches} branch nets ({CFG['input_dim']} inputs × {CFG['n_outputs'] if 'n_outputs' in CFG else 2} channels)")
print(f"   branch={CFG['branch_depth']}×{CFG['branch_width']} (each sees 1 scalar) | trunk={CFG['trunk_depth']}×{CFG['trunk_width']}")
print(f"   lr={CFG['lr']} | wd={CFG['weight_decay']} | batch={CFG['batch_size']}")


# ============================================================
# Load & scale training data
# ============================================================
X_tr, Yt_tr, Ys_tr = load_split(CFG["train_dir"], Nx, Nt)

x_scaler      = StandardScaler()
X_tr_scaled   = x_scaler.fit_transform(X_tr)

temp_scaler   = StandardScaler()
Yt_tr_s       = temp_scaler.fit_transform(Yt_tr.reshape(len(X_tr), -1)).reshape(len(X_tr), Nx, Nt)

stress_scaler = StandardScaler()
Ys_tr_s       = stress_scaler.fit_transform(Ys_tr.reshape(len(X_tr), -1)).reshape(len(X_tr), Nx, Nt)

Y_tr_combined = np.stack([Yt_tr_s, Ys_tr_s], axis=1)   # (N, 2, Nx, Nt)

# Save scalers
np.save(os.path.join(save_dir, "x_scaler_mean.npy"),    x_scaler.mean_)
np.save(os.path.join(save_dir, "x_scaler_scale.npy"),   x_scaler.scale_)
np.save(os.path.join(save_dir, "y_temp_mean.npy"),      temp_scaler.mean_.reshape(Nx, Nt))
np.save(os.path.join(save_dir, "y_temp_scale.npy"),     temp_scaler.scale_.reshape(Nx, Nt))
np.save(os.path.join(save_dir, "y_stress_mean.npy"),    stress_scaler.mean_.reshape(Nx, Nt))
np.save(os.path.join(save_dir, "y_stress_scale.npy"),   stress_scaler.scale_.reshape(Nx, Nt))


# ============================================================
# Build model + optimizer
# ============================================================
model = MIONet(
    input_dim    = CFG["input_dim"],
    p            = CFG["p"],
    branch_width = CFG["branch_width"],
    branch_depth = CFG["branch_depth"],
    trunk_width  = CFG["trunk_width"],
    trunk_depth  = CFG["trunk_depth"],
    n_outputs    = 2,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ MIONet | trainable params: {total_params:,}\n")

opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, factor=0.5, patience=200, threshold=1e-4
)


# ============================================================
# Prepare tensors
# ============================================================
X_train = torch.tensor(X_tr_scaled,   dtype=torch.float32, device=device)
Y_train = torch.tensor(Y_tr_combined, dtype=torch.float32, device=device)

XT = torch.tensor(make_xt_grid(Nx, Nt), dtype=torch.float32, device=device)  # (Nx*Nt, 2)

N          = X_train.shape[0]
batch_size = min(CFG["batch_size"], N)


# ============================================================
# Training loop
# ============================================================
sync(device)
t0        = time.time()
best_loss = float("inf")

for epoch in range(1, CFG["epochs"] + 1):
    model.train()
    perm       = torch.randperm(N, device=device)
    epoch_loss = 0.0
    loss_T_ep  = 0.0
    loss_S_ep  = 0.0

    for i in range(0, N, batch_size):
        idx  = perm[i:i+batch_size]
        xb   = X_train[idx]
        yb   = Y_train[idx]

        opt.zero_grad(set_to_none=True)
        pred = model(xb, XT, Nx, Nt)   # (B, 2, Nx, Nt)

        loss_T = F.mse_loss(pred[:, 0], yb[:, 0])
        loss_S = F.mse_loss(pred[:, 1], yb[:, 1])
        loss   = CFG["wT"] * loss_T + CFG["wS"] * loss_S

        loss.backward()
        opt.step()

        bs         = xb.shape[0]
        epoch_loss += loss.item()   * bs
        loss_T_ep  += loss_T.item() * bs
        loss_S_ep  += loss_S.item() * bs

    epoch_loss /= N
    loss_T_ep  /= N
    loss_S_ep  /= N
    sched.step(epoch_loss)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_path)

    if epoch % 100 == 0 or epoch == 1:
        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch:4d} | Loss={epoch_loss:.6e} "
              f"(T={loss_T_ep:.3e}, S={loss_S_ep:.3e}) | lr={lr_now:.2e}")

sync(device)
train_time = time.time() - t0

torch.save(model.state_dict(), weights_path)
print(f"\n⏱️  Training time: {train_time:.2f}s")
print(f"✅ Saved LAST weights → {weights_path}")
print(f"🏁 Saved BEST weights → {best_path}  (best_loss={best_loss:.6e})")


# ============================================================
# Evaluation helper
# ============================================================
def evaluate_split(split_name: str, split_dir: str, use_best: bool = True):
    wp = best_path if (use_best and os.path.exists(best_path)) else weights_path
    model.load_state_dict(torch.load(wp, map_location=device))
    model.eval()

    X, Yt, Ys = load_split(split_dir, Nx, Nt)
    X_scaled  = (X - x_scaler.mean_) / x_scaler.scale_

    bs    = min(CFG["infer_batch_size"], len(X_scaled))
    preds = []

    # warmup
    with torch.inference_mode():
        xb0 = torch.tensor(X_scaled[:bs], dtype=torch.float32, device=device)
        _ = model(xb0, XT, Nx, Nt)
        sync(device)

    sync(device)
    t_inf = time.time()
    with torch.inference_mode():
        for i in range(0, len(X_scaled), bs):
            xb  = torch.tensor(X_scaled[i:i+bs], dtype=torch.float32, device=device)
            out = model(xb, XT, Nx, Nt)
            preds.append(out.detach().cpu())
    sync(device)
    infer_time = time.time() - t_inf

    Y_pred_s = torch.cat(preds, dim=0).numpy()   # (N, 2, Nx, Nt)

    temp_mean   = temp_scaler.mean_.reshape(Nx, Nt)
    temp_std    = temp_scaler.scale_.reshape(Nx, Nt)
    stress_mean = stress_scaler.mean_.reshape(Nx, Nt)
    stress_std  = stress_scaler.scale_.reshape(Nx, Nt)

    Y_pred_temp   = Y_pred_s[:, 0] * temp_std   + temp_mean
    Y_pred_stress = Y_pred_s[:, 1] * stress_std + stress_mean

    temp_l2   = rel_l2(Y_pred_temp,   Yt)
    stress_l2 = rel_l2(Y_pred_stress, Ys)

    print(f"\n📊 ====== MIONet Evaluation ({split_name}) ======")
    print(f"L2(T): {temp_l2:.6e} | L2(σ): {stress_l2:.6e}  "
          f"({temp_l2*100:.3f}%, {stress_l2*100:.3f}%)")
    print(f"⏱️  Inference: {infer_time:.3f}s for {len(X_scaled)} samples "
          f"({infer_time/max(len(X_scaled),1):.6f}s/sample)")

    metrics = {
        "model":                      "MIONet",
        "split":                      split_name,
        "device":                     device.type,
        "num_samples":                int(len(X_scaled)),
        "training_time_s":            float(train_time),
        "inference_time_total_s":     float(infer_time),
        "inference_time_per_sample_s":float(infer_time / max(len(X_scaled), 1)),
        "l2_temp":                    float(temp_l2),
        "l2_stress":                  float(stress_l2),
        "l2_temp_percent":            float(temp_l2   * 100),
        "l2_stress_percent":          float(stress_l2 * 100),
        "p":                          int(CFG["p"]),
        "n_branch_nets":              int(CFG["input_dim"] * 2),
        "use_best":                   bool(use_best),
    }

    out_json = os.path.join(save_dir, f"metrics_{split_name}_{device.type}.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Saved metrics → {out_json}")


evaluate_split("train",        CFG["train_dir"], use_best=True)
evaluate_split("test_unseen",  CFG["test_dir"],  use_best=True)
