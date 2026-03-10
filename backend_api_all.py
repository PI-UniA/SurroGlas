# backend_api_all.py
# ------------------------------------------------------------
# Unified FastAPI backend for SurroGlas models
# Supports: MIFNO (multiscale), MIONet (multiplicative fusion), MLP (optional)
# Device: CPU / MPS / CUDA via SURROGLAS_DEVICE=auto|cpu|mps|cuda
# ------------------------------------------------------------

import os, json
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2, irfft2
import time

# ----------------- Optional Dependencies -----------------
try:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    HAVE_JAX = True
except Exception:
    HAVE_JAX = False

try:
    from MLP.mlp_model_template import MLP
    HAVE_MLP = True and HAVE_JAX
except Exception:
    HAVE_MLP = False

# ----------------- Global Config -----------------
Nx, Nt = 49, 500

def pick_torch_device(req: str) -> torch.device:
    req = (req or "auto").lower().strip()
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

SURROGLAS_DEVICE = os.environ.get("SURROGLAS_DEVICE", "auto")
device = pick_torch_device(SURROGLAS_DEVICE)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

app = FastAPI(
    title="SurroGlas Backend API",
    version="2.0.0",
    description=(
        "FastAPI backend serving MIFNO (multiscale input FNO), "
        "MIONet (multiplicative multi-input operator), and MLP models "
        "for glass cooling predictions."
    )
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
#  TRUE MIFNO ARCHITECTURE
#  Mirrors MIFNO/train_MIFNO.py exactly — must stay in sync.
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
                nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, nx_k * nt_k))
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
            field_k = self.scale_lifters[k](x_scalar).view(B, 1, nx_k, nt_k)
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
        return F.gelu(self.norm(self.spec(x) + self.lin(x) + self.cond_proj(cond)))


class MIFNO(nn.Module):
    def __init__(self, Nx, Nt, param_dim=4, n_scales=3, scale_width=16,
                 width=64, depth=6, modes_x=16, modes_t=16):
        super().__init__()
        self.param_dim = param_dim
        self.input_encoders = nn.ModuleList([
            MultiscaleInputEncoder(Nx, Nt, n_scales, scale_width)
            for _ in range(param_dim)
        ])
        in_ch = param_dim * n_scales * scale_width
        self.fusion = CrossScaleFusion(in_ch, width, Nx, Nt)
        self.blocks = nn.ModuleList([
            MIFNOBlock(width, width, modes_x, modes_t) for _ in range(depth)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(width // 2, 2, kernel_size=1),
        )

    def forward(self, x_vec: torch.Tensor) -> torch.Tensor:
        per_input = [enc(x_vec[:, i:i+1]) for i, enc in enumerate(self.input_encoders)]
        cond = self.fusion(torch.cat(per_input, dim=1))
        x = cond
        for blk in self.blocks:
            x = blk(x, cond)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════
#  TRUE MIONet ARCHITECTURE
#  Mirrors MIONET/train_eval_MIONET_operator.py exactly.
# ═══════════════════════════════════════════════════════════════

def _make_mlp(in_dim: int, width: int, depth: int, out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.GELU()]
        d = width
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

def _make_xt_grid(Nx: int, Nt: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, Nx, dtype=np.float32)
    t = np.linspace(0.0, 1.0, Nt, dtype=np.float32)
    Xg, Tg = np.meshgrid(x, t, indexing="ij")
    return np.stack([Xg, Tg], axis=-1).reshape(-1, 2)

class MIONet(nn.Module):
    def __init__(self, input_dim=4, p=128, branch_width=128, branch_depth=3,
                 trunk_width=256, trunk_depth=4, n_outputs=2):
        super().__init__()
        self.p         = p
        self.input_dim = input_dim
        self.n_outputs = n_outputs
        self.trunk = _make_mlp(2, trunk_width, trunk_depth, p)
        self.branches = nn.ModuleList([
            nn.ModuleList([
                _make_mlp(1, branch_width, branch_depth, p)
                for _ in range(input_dim)
            ])
            for _ in range(n_outputs)
        ])
        self.bias = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, x_params: torch.Tensor, xt: torch.Tensor,
                Nx: int, Nt: int) -> torch.Tensor:
        B   = x_params.shape[0]
        phi = self.trunk(xt)                          # (P, p)
        channel_outputs = []
        for c in range(self.n_outputs):
            fused = torch.ones(B, self.p, device=x_params.device)
            for i, branch in enumerate(self.branches[c]):
                fused = fused * branch(x_params[:, i:i+1])
            out_flat = fused @ phi.T + self.bias[c]   # (B, P)
            channel_outputs.append(out_flat.view(B, Nx, Nt))
        return torch.stack(channel_outputs, dim=1)    # (B, 2, Nx, Nt)


# ----------------- Helpers -----------------
def _must_load(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)

def _weights_path(base_dir: str, stem: str) -> str:
    for suffix in [f"_{device.type}_best.pt", f"_{device.type}.pt", "_best.pt", ".pt"]:
        p = os.path.join(base_dir, stem + suffix)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No weights found for stem '{stem}' in '{base_dir}' "
        f"(tried suffixes: _{{device}}_best.pt, _{{device}}.pt, _best.pt, .pt)"
    )

def _load_cfg(path: str, defaults: dict) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        print(f"   config loaded from {path}")
        return {**defaults, **cfg}
    print(f"   ⚠️  config not found at {path}, using defaults")
    return defaults

MODELS:  Dict[str, Any]                  = {}
SCALERS: Dict[str, Dict[str, np.ndarray]] = {}
# MIONet needs the coordinate grid at inference time
XT_GRID: Optional[torch.Tensor]           = None


# ----------------- Schemas -----------------
class PredictRequest(BaseModel):
    model:   str   = Field(..., description="Model name: mifno, mionet, or mlp")
    htc:     float = Field(..., description="Convective heat transfer coefficient")
    epsilon: float = Field(..., ge=0.0, le=1.0, description="Surface emissivity (0–1)")
    sigma:   float = Field(..., gt=0.0, description="Stefan–Boltzmann constant or effective value")
    alpha:   float = Field(..., gt=0.0, description="Thermal diffusivity")

class PredictBatchRequest(BaseModel):
    model: str
    X: List[List[float]]   # list of [htc, epsilon, sigma, alpha]

class PredictResponse(BaseModel):
    temperature:       List[List[float]]
    stress:            List[List[float]]
    inference_time_ms: float
    device:            str


# ----------------- Startup: Load Models -----------------
@app.on_event("startup")
def _startup_load_models():
    global XT_GRID
    print(f"⚙️  Backend starting | torch_device={device} | version=2.0.0")

    # ── MIFNO ────────────────────────────────────────────────
    try:
        mifno_defaults = dict(
            Nx=Nx, Nt=Nt, param_dim=4, n_scales=3, scale_width=16,
            width=64, depth=6, modes_x=16, modes_t=16
        )
        cfg = _load_cfg("MIFNO/mifno_config.json", mifno_defaults)

        mifno = MIFNO(
            Nx=int(cfg["Nx"]), Nt=int(cfg["Nt"]),
            param_dim=int(cfg["param_dim"]),
            n_scales=int(cfg["n_scales"]),
            scale_width=int(cfg["scale_width"]),
            width=int(cfg["width"]),
            depth=int(cfg["depth"]),
            modes_x=int(cfg["modes_x"]),
            modes_t=int(cfg["modes_t"]),
        )
        wpath = _weights_path("MIFNO", "mifno_model_dual")
        mifno.load_state_dict(torch.load(wpath, map_location="cpu"), strict=True)
        mifno.eval().to(device)
        MODELS["mifno"] = mifno
        SCALERS["mifno"] = {
            "x_mean":        _must_load("MIFNO/x_scaler_mean_mifno.npy"),
            "x_std":         _must_load("MIFNO/x_scaler_scale_mifno.npy"),
            "y_temp_mean":   _must_load("MIFNO/y_temp_scaler_mean.npy").reshape(Nx, Nt),
            "y_temp_std":    _must_load("MIFNO/y_temp_scaler_scale.npy").reshape(Nx, Nt),
            "y_stress_mean": _must_load("MIFNO/y_stress_scaler_mean.npy").reshape(Nx, Nt),
            "y_stress_std":  _must_load("MIFNO/y_stress_scaler_scale.npy").reshape(Nx, Nt),
        }
        print(f"✅ MIFNO loaded  ({wpath})")
    except Exception as e:
        print(f"⚠️  Failed to load MIFNO: {e}")

    # ── MIONet ───────────────────────────────────────────────
    try:
        mionet_defaults = dict(
            input_dim=4, p=128, branch_width=128, branch_depth=3,
            trunk_width=256, trunk_depth=4
        )
        cfg = _load_cfg("MIONET/config.json", mionet_defaults)

        mionet = MIONet(
            input_dim=int(cfg["input_dim"]),
            p=int(cfg["p"]),
            branch_width=int(cfg["branch_width"]),
            branch_depth=int(cfg["branch_depth"]),
            trunk_width=int(cfg["trunk_width"]),
            trunk_depth=int(cfg["trunk_depth"]),
            n_outputs=2,
        )
        wpath = _weights_path("MIONET", "mionet_operator_dual")
        mionet.load_state_dict(torch.load(wpath, map_location="cpu"), strict=True)
        mionet.eval().to(device)
        MODELS["mionet"] = mionet
        SCALERS["mionet"] = {
            "x_mean":        _must_load("MIONET/x_scaler_mean.npy"),
            "x_std":         _must_load("MIONET/x_scaler_scale.npy"),
            "y_temp_mean":   _must_load("MIONET/y_temp_mean.npy").reshape(Nx, Nt),
            "y_temp_std":    _must_load("MIONET/y_temp_scale.npy").reshape(Nx, Nt),
            "y_stress_mean": _must_load("MIONET/y_stress_mean.npy").reshape(Nx, Nt),
            "y_stress_std":  _must_load("MIONET/y_stress_scale.npy").reshape(Nx, Nt),
        }
        # pre-build coordinate grid on device (shared across all MIONet calls)
        XT_GRID = torch.tensor(_make_xt_grid(Nx, Nt), dtype=torch.float32, device=device)
        print(f"✅ MIONet loaded ({wpath})")
    except Exception as e:
        print(f"⚠️  Failed to load MIONet: {e}")

    # ── MLP (JAX/Equinox) ────────────────────────────────────
    if HAVE_MLP:
        try:
            mlp_template = MLP(in_dim=4, out_dim=2 * Nx * Nt, width=512, depth=4,
                               key=jax.random.PRNGKey(0))
            mlp_model = eqx.tree_deserialise_leaves("MLP/mlp_model_joint.eqx", mlp_template)
            MODELS["mlp"] = mlp_model
            SCALERS["mlp"] = {
                "x_mean":        _must_load("MLP/x_scaler_mean_mlp.npy"),
                "x_std":         _must_load("MLP/x_scaler_scale_mlp.npy"),
                "y_temp_mean":   _must_load("MLP/y_temp_scaler_mean.npy").reshape(Nx, Nt),
                "y_temp_std":    _must_load("MLP/y_temp_scaler_scale.npy").reshape(Nx, Nt),
                "y_stress_mean": _must_load("MLP/y_stress_scaler_mean.npy").reshape(Nx, Nt),
                "y_stress_std":  _must_load("MLP/y_stress_scaler_scale.npy").reshape(Nx, Nt),
            }
            print("✅ MLP loaded")
        except Exception as e:
            print(f"⚠️  Failed to load MLP: {e}")

    print(f"🚀 Ready | loaded models: {list(MODELS.keys())}")


# ----------------- Inference utilities -----------------
def _scale_inputs(model_name: str, X: np.ndarray) -> np.ndarray:
    return (X - SCALERS[model_name]["x_mean"]) / SCALERS[model_name]["x_std"]

def _inverse_scale(model_name: str, y_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    sc = SCALERS[model_name]
    return {
        "temperature": y_scaled[:, 0] * sc["y_temp_std"]   + sc["y_temp_mean"],
        "stress":      y_scaled[:, 1] * sc["y_stress_std"] + sc["y_stress_mean"],
    }

def _forward_mifno(model: MIFNO, X_scaled: np.ndarray) -> tuple[np.ndarray, float]:
    with torch.inference_mode():
        xb = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        # warmup
        _ = model(xb[:1]); sync(device)
        sync(device); t0 = time.perf_counter()
        out = model(xb)
        sync(device); ms = (time.perf_counter() - t0) * 1000.0
        return out.detach().cpu().numpy(), ms

def _forward_mionet(model: MIONet, X_scaled: np.ndarray) -> tuple[np.ndarray, float]:
    assert XT_GRID is not None, "XT_GRID not initialised — MIONet failed to load."
    with torch.inference_mode():
        xb = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        # warmup
        _ = model(xb[:1], XT_GRID, Nx, Nt); sync(device)
        sync(device); t0 = time.perf_counter()
        out = model(xb, XT_GRID, Nx, Nt)
        sync(device); ms = (time.perf_counter() - t0) * 1000.0
        return out.detach().cpu().numpy(), ms

def _forward_mlp(mlp_model: Any, X_scaled: np.ndarray) -> tuple[np.ndarray, float]:
    xj = jnp.array(X_scaled)
    t0 = time.perf_counter()
    y_flat = jax.vmap(mlp_model)(xj)
    ms = (time.perf_counter() - t0) * 1000.0
    return np.array(y_flat).reshape(-1, 2, Nx, Nt), ms

def _run_model(model_name: str, X_scaled: np.ndarray) -> tuple[np.ndarray, float]:
    """Dispatch to the correct forward function based on model type."""
    if model_name == "mifno":
        return _forward_mifno(MODELS["mifno"], X_scaled)
    if model_name == "mionet":
        return _forward_mionet(MODELS["mionet"], X_scaled)
    if model_name == "mlp":
        if not HAVE_MLP:
            raise HTTPException(status_code=400, detail="MLP/JAX backend not available.")
        return _forward_mlp(MODELS["mlp"], X_scaled)
    raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")


# ----------------- Endpoints -----------------
@app.get("/health")
def health():
    return {
        "status":      "ok",
        "version":     "2.0.0",
        "Nx":          Nx,
        "Nt":          Nt,
        "models":      list(MODELS.keys()),
        "device":      device.type,
        "torch":       torch.__version__,
        "cuda_avail":  bool(torch.cuda.is_available()),
        "mps_avail":   bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "mifno_arch":  "MultiscaleInputEncoder + CrossScaleFusion + MIFNOBlock(cond_injection)",
        "mionet_arch": "4×2 independent branch MLPs (⊙ fusion) + shared trunk + dot-product",
    }

@app.post("/predict", response_model=PredictResponse)
def predict_fields(req: PredictRequest):
    model_name = req.model.lower()
    if model_name not in MODELS:
        raise HTTPException(status_code=400,
                            detail=f"Model '{model_name}' not loaded. Available: {list(MODELS.keys())}")

    X       = np.array([[req.htc, req.epsilon, req.sigma, req.alpha]], dtype=np.float32)
    Xs      = _scale_inputs(model_name, X)
    y_s, ms = _run_model(model_name, Xs)
    inv     = _inverse_scale(model_name, y_s)

    return PredictResponse(
        temperature       = inv["temperature"][0].tolist(),
        stress            = inv["stress"][0].tolist(),
        inference_time_ms = float(ms),
        device            = device.type,
    )

@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    model_name = req.model.lower()
    if model_name not in MODELS:
        raise HTTPException(status_code=400,
                            detail=f"Model '{model_name}' not loaded. Available: {list(MODELS.keys())}")
    if not req.X or any((not isinstance(r, list) or len(r) != 4) for r in req.X):
        raise HTTPException(status_code=400,
                            detail="X must be a list of [htc, epsilon, sigma, alpha] rows.")

    X       = np.asarray(req.X, dtype=np.float32)
    Xs      = _scale_inputs(model_name, X)
    y_s, ms = _run_model(model_name, Xs)
    inv     = _inverse_scale(model_name, y_s)
    B       = int(X.shape[0])

    return {
        "temperature":                inv["temperature"].tolist(),
        "stress":                     inv["stress"].tolist(),
        "B":                          B,
        "Nx":                         Nx,
        "Nt":                         Nt,
        "model":                      model_name,
        "device":                     device.type,
        "inference_time_ms":          float(ms),
        "inference_time_ms_per_sample": float(ms / max(B, 1)),
    }


# ----------------- Local run -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api_all:app", host="0.0.0.0", port=8000, reload=True)
