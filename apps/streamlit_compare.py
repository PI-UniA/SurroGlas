import os
import re
import json
import time
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.fft import rfft2, irfft2


# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Annealing Lehr Comparison", layout="wide")
st.title("Annealing Lehr: FEM vs MIFNO vs MIONet")


# ============================================================
# Lehr zones
# ============================================================
ZONE_TIME_WINDOWS = {
    "A1": (0.0, 54.7),
    "A2": (54.7, 109.6),
    "B1": (109.6, 182.6),
    "B2": (182.6, 255.57),
    "C1": (255.57, 328.5),
}

ZONE_COLORS = {
    "A1": "#dbeafe",
    "A2": "#dcfce7",
    "B1": "#fef3c7",
    "B2": "#fde2e2",
    "C1": "#ede9fe",
}


# ============================================================
# Helpers
# ============================================================
def pick_device(req: str) -> torch.device:
    req = req.lower().strip()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if req == "auto":
        return torch.device("cpu")
    return torch.device("cpu")


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def extract_case_idx(fname: str) -> int:
    m = re.search(r"case[_]?(\d+)|case(\d+)", fname)
    if m:
        for g in m.groups():
            if g is not None:
                return int(g)
    return 10**9


def rel_l2(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)


# --- Helper: build zone-wise param vector from sidebar ---
def build_param_vector_from_sidebar(param_names, sidebar_values):
    vals = []
    for name in param_names:
        if name not in sidebar_values:
            raise KeyError(f"Missing sidebar value for parameter '{name}'")
        vals.append(float(sidebar_values[name]))
    return np.array(vals, dtype=np.float32)


# --- Helper: stress vs distance plot ---
def plot_stress_vs_distance(distance_axis, fem_stress, mifno_stress, mionet_stress,
                            surface_idx, mid_idx, velocity, zone_windows,
                            zone_colors, show_zones=True, zone_alpha=0.35,
                            show_fem=True):
    fig, ax = plt.subplots(figsize=(8, 4))

    if show_fem:
        ax.plot(distance_axis, fem_stress[surface_idx, :], "--", label="FEM surface")
    ax.plot(distance_axis, mifno_stress[surface_idx, :], label="MIFNO surface")
    ax.plot(distance_axis, mionet_stress[surface_idx, :], label="MIONet surface")

    if show_fem:
        ax.plot(distance_axis, fem_stress[mid_idx, :], "--", label="FEM mid")
    ax.plot(distance_axis, mifno_stress[mid_idx, :], label="MIFNO mid")
    ax.plot(distance_axis, mionet_stress[mid_idx, :], label="MIONet mid")

    ax.set_xlabel("Lehr distance x (m)")
    ax.set_ylabel("Stress (Pa)")
    ax.set_title("Stress vs distance", fontsize=14, fontweight="semibold", pad=18)
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)

    if show_zones:
        add_zone_background_distance(ax, distance_axis, velocity, zone_windows, zone_colors, alpha=zone_alpha)
    return fig


# --- Helper: through-thickness stress at t ---
def plot_through_thickness_stress_at_t(fem_stress, mifno_stress, mionet_stress, t_idx, show_fem=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    if show_fem:
        ax.plot(fem_stress[:, t_idx], "--", label="FEM")
    ax.plot(mifno_stress[:, t_idx], label="MIFNO")
    ax.plot(mionet_stress[:, t_idx], label="MIONet")
    ax.set_xlabel("Thickness index")
    ax.set_ylabel("Stress (Pa)")
    ax.set_title(f"Through-thickness stress at t_idx={t_idx}", fontsize=14, fontweight="semibold", pad=18)
    ax.grid(True)
    ax.legend()
    return fig

# --- Helper: field map plot ---
def plot_field_map(field, title, cbar_label, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(field, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Thickness index")
    fig.colorbar(im, ax=ax).set_label(cbar_label)
    return fig


# --- Helper: error map plot ---
def plot_error_map(pred_field, true_field, title, cbar_label):
    err = np.abs(pred_field - true_field)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(err, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Thickness index")
    fig.colorbar(im, ax=ax).set_label(cbar_label)
    return fig


def add_zone_background_time(ax, t_axis, zone_windows, zone_colors, alpha=0.35):
    t_min, t_max = t_axis[0], t_axis[-1]
    for zone, (t0, t1) in zone_windows.items():
        left = max(t0, t_min)
        right = min(t1, t_max)
        if right > left:
            ax.axvspan(left, right, color=zone_colors.get(zone, "lightgray"), alpha=alpha, zorder=0)
            center = 0.5 * (left + right)
            ax.text(center, 1.02, zone, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9)
        if t_min <= t1 <= t_max:
            ax.axvline(t1, linestyle="--", linewidth=1, color="gray")


def add_zone_background_distance(ax, x_axis, velocity, zone_windows, zone_colors, alpha=0.35):
    x_min, x_max = x_axis[0], x_axis[-1]
    for zone, (t0, t1) in zone_windows.items():
        x0 = velocity * t0
        x1 = velocity * t1
        left = max(x0, x_min)
        right = min(x1, x_max)
        if right > left:
            ax.axvspan(left, right, color=zone_colors.get(zone, "lightgray"), alpha=alpha, zorder=0)
            center = 0.5 * (left + right)
            ax.text(center, 1.02, zone, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9)
        if x_min <= x1 <= x_max:
            ax.axvline(x1, linestyle="--", linewidth=1, color="gray")

def build_zone_kpi_table(temp_field, stress_field, t_axis, zone_windows, surface_idx=0, mid_idx=None):
    if mid_idx is None:
        mid_idx = temp_field.shape[0] // 2

    rows = []
    for zone, (t0, t1) in zone_windows.items():
        idx = np.where((t_axis >= t0) & (t_axis < t1))[0]
        if len(idx) == 0:
            continue

        temp_zone = temp_field[:, idx]
        stress_zone = stress_field[:, idx]

        rows.append({
            "Zone": zone,
            "t_start_s": float(t0),
            "t_end_s": float(t1),
            "Mean surface T (K)": float(np.mean(temp_zone[surface_idx, :])),
            "Exit surface T (K)": float(temp_zone[surface_idx, -1]),
            "Mean mid-plane T (K)": float(np.mean(temp_zone[mid_idx, :])),
            "Max stress (Pa)": float(np.max(stress_zone)),
            "Min stress (Pa)": float(np.min(stress_zone)),
            "Exit surface stress (Pa)": float(stress_zone[surface_idx, -1]),
            "Exit mid stress (Pa)": float(stress_zone[mid_idx, -1]),
        })

    return rows
# ============================================================
# MIFNO
# ============================================================
class MultiscaleInputEncoder(nn.Module):
    def __init__(self, Nx, Nt, n_scales, scale_width):
        super().__init__()
        self.Nx, self.Nt = Nx, Nt
        self.n_scales = n_scales

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
        self.weight = nn.Parameter(
            torch.randn(in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat) / (in_ch * out_ch)
        )

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
        self.blocks = nn.ModuleList([MIFNOBlock(width, width, modes_x, modes_t) for _ in range(depth)])
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
# MIONet
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
# Cached loaders
# ============================================================
@st.cache_resource
def load_mifno(device_str: str):
    device = pick_device(device_str)
    save_dir = "MIFNO"
    with open(os.path.join(save_dir, "latest_config.json")) as f:
        cfg = json.load(f)

    model = MIFNO(
        Nx=cfg["Nx"], Nt=cfg["Nt"], param_dim=cfg["param_dim"],
        n_scales=cfg["n_scales"], scale_width=cfg["scale_width"],
        width=cfg["width"], depth=cfg["depth"],
        modes_x=cfg["modes_x"], modes_t=cfg["modes_t"],
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, "latest_model_best.pt"), map_location=device), strict=True)
    model.eval()

    scalers = {
        "x_mean": np.load(os.path.join(save_dir, "x_scaler_mean_mifno_mz.npy")),
        "x_std": np.load(os.path.join(save_dir, "x_scaler_scale_mifno_mz.npy")),
        "yT_mean": np.load(os.path.join(save_dir, "y_temp_scaler_mean_mz.npy")),
        "yT_std": np.load(os.path.join(save_dir, "y_temp_scaler_scale_mz.npy")),
        "yS_mean": np.load(os.path.join(save_dir, "y_stress_scaler_mean_mz.npy")),
        "yS_std": np.load(os.path.join(save_dir, "y_stress_scaler_scale_mz.npy")),
    }
    return cfg, model, scalers


@st.cache_resource
def load_mionet(device_str: str):
    device = pick_device(device_str)
    save_dir = "MIONET"
    with open(os.path.join(save_dir, "latest_config_mionet.json")) as f:
        cfg = json.load(f)

    model = MIONet(
        param_dim=cfg["param_dim"],
        latent_dim=cfg["latent_dim"],
        branch_width=cfg["branch_width"],
        branch_depth=cfg["branch_depth"],
        trunk_width=cfg["trunk_width"],
        trunk_depth=cfg["trunk_depth"],
        Nx=cfg["Nx"], Nt=cfg["Nt"],
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, "latest_model_best_mionet.pt"), map_location=device), strict=True)
    model.eval()

    scalers = {
        "x_mean": np.load(os.path.join(save_dir, "x_scaler_mean_mionet_mz.npy")),
        "x_std": np.load(os.path.join(save_dir, "x_scaler_scale_mionet_mz.npy")),
        "yT_mean": np.load(os.path.join(save_dir, "y_temp_scaler_mean_mionet_mz.npy")),
        "yT_std": np.load(os.path.join(save_dir, "y_temp_scaler_scale_mionet_mz.npy")),
        "yS_mean": np.load(os.path.join(save_dir, "y_stress_scaler_mean_mionet_mz.npy")),
        "yS_std": np.load(os.path.join(save_dir, "y_stress_scaler_scale_mionet_mz.npy")),
    }
    return cfg, model, scalers


@st.cache_data
def load_dataset(split: str, Nt: int, Nx: int):
    data_dir = os.path.join("results", split)
    param_files = sorted([f for f in os.listdir(data_dir) if f.startswith("params_case_")], key=extract_case_idx)
    temp_files = sorted([f for f in os.listdir(data_dir) if f.startswith("temperature_all_case")], key=extract_case_idx)
    stress_files = sorted([f for f in os.listdir(data_dir) if f.startswith("stress_all_case")], key=extract_case_idx)

    X, Y_temp, Y_stress = [], [], []
    for p, t, s in zip(param_files, temp_files, stress_files):
        X.append(np.loadtxt(os.path.join(data_dir, p), delimiter=",", skiprows=1))
        Y_temp.append(np.loadtxt(os.path.join(data_dir, t)))
        Y_stress.append(np.loadtxt(os.path.join(data_dir, s)))

    X = np.array(X).astype(np.float32)
    Y_temp = np.array(Y_temp).reshape(len(X), Nt, Nx).transpose(0, 2, 1)
    Y_stress = np.array(Y_stress).reshape(len(X), Nt, Nx).transpose(0, 2, 1)
    return X, Y_temp, Y_stress

@st.cache_data
def load_fem_timing_summary(split: str):
    summary_path = os.path.join("results", split, "timing_summary.json")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r") as f:
        return json.load(f)

# ============================================================
# Sidebar
# ============================================================
mode = st.sidebar.radio(
    "Mode",
    ["Dataset evaluation", "Manual prediction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Configuration")

device_choice = st.sidebar.selectbox("Device", ["cpu", "auto", "cuda"], index=0)
dt = st.sidebar.number_input("dt (s)", value=0.1, step=0.01, format="%.3f")
velocity = st.sidebar.number_input("Velocity (m/s)", value=0.24, step=0.01, format="%.3f")
show_zones = st.sidebar.checkbox("Show Lehr zones", value=True)
zone_alpha = st.sidebar.slider("Zone shading", 0.0, 0.8, 0.35, 0.05)

mifno_cfg, mifno_model, mifno_scalers = load_mifno(device_choice)
mionet_cfg, mionet_model, mionet_scalers = load_mionet(device_choice)

Nx = mifno_cfg["Nx"]
Nt = mifno_cfg["Nt"]
device = pick_device(device_choice)

if mode == "Dataset evaluation":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset selection")
    split = st.sidebar.selectbox("Dataset split", ["test_unseen", "train"], index=0)

    X, Y_temp, Y_stress = load_dataset(split, Nt, Nx)
    timing_summary = load_fem_timing_summary(split)

    case_id = st.sidebar.slider("Case ID", 0, len(X) - 1, 0)
    t_idx = st.sidebar.slider("Time index", 0, Nt - 1, min(190, Nt - 1))

    use_manual_params = False

else:
    split = "test_unseen"  # only used as reference dataset for FEM overlay if needed
    X, Y_temp, Y_stress = load_dataset(split, Nt, Nx)
    timing_summary = load_fem_timing_summary(split)

    case_id = st.sidebar.slider("Reference FEM case", 0, len(X) - 1, 0)
    t_idx = st.sidebar.slider("Time index", 0, Nt - 1, min(190, Nt - 1))

    mifno_param_names = mifno_cfg.get("param_names", [])

    def case_param_default(name: str, fallback: float) -> float:
        if name in mifno_param_names:
            idx = mifno_param_names.index(name)
            return float(X[case_id, idx])
        return float(fallback)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Manual zone-wise inputs")

    htc_A1 = st.sidebar.number_input("htc_A1", value=case_param_default("htc_A1", 30.0), step=1.0)
    htc_A2 = st.sidebar.number_input("htc_A2", value=case_param_default("htc_A2", 30.0), step=1.0)
    htc_B1 = st.sidebar.number_input("htc_B1", value=case_param_default("htc_B1", 30.0), step=1.0)
    htc_B2 = st.sidebar.number_input("htc_B2", value=case_param_default("htc_B2", 30.0), step=1.0)
    htc_C1 = st.sidebar.number_input("htc_C1", value=case_param_default("htc_C1", 30.0), step=1.0)

    T_amb_A1 = st.sidebar.number_input("T_amb_A1", value=case_param_default("T_amb_A1", 873.0), step=1.0)
    T_amb_A2 = st.sidebar.number_input("T_amb_A2", value=case_param_default("T_amb_A2", 853.0), step=1.0)
    T_amb_B1 = st.sidebar.number_input("T_amb_B1", value=case_param_default("T_amb_B1", 827.0), step=1.0)
    T_amb_B2 = st.sidebar.number_input("T_amb_B2", value=case_param_default("T_amb_B2", 779.0), step=1.0)
    T_amb_C1 = st.sidebar.number_input("T_amb_C1", value=case_param_default("T_amb_C1", 757.0), step=1.0)

    epsilon = st.sidebar.number_input("epsilon", value=case_param_default("epsilon", 0.85), step=0.01, format="%.4f")
    sigma = st.sidebar.number_input("sigma", value=case_param_default("sigma", 5.67e-08), format="%.6e")

    sidebar_values = {
        "htc_A1": htc_A1,
        "htc_A2": htc_A2,
        "htc_B1": htc_B1,
        "htc_B2": htc_B2,
        "htc_C1": htc_C1,
        "T_amb_A1": T_amb_A1,
        "T_amb_A2": T_amb_A2,
        "T_amb_B1": T_amb_B1,
        "T_amb_B2": T_amb_B2,
        "T_amb_C1": T_amb_C1,
        "epsilon": epsilon,
        "sigma": sigma,
    }

    X_manual_mifno = build_param_vector_from_sidebar(mifno_param_names, sidebar_values).reshape(1, -1)
    mionet_param_names = mionet_cfg.get("param_names", mifno_param_names)
    X_manual_mionet = build_param_vector_from_sidebar(mionet_param_names, sidebar_values).reshape(1, -1)

    use_manual_params = True

t_axis = np.arange(Nt) * dt
x_axis = velocity * t_axis
surface_idx = 0
mid_idx = Nx // 2

# MIONet coords
x_coords = np.linspace(0.0, 1.0, Nx, dtype=np.float32)
t_coords = np.linspace(0.0, 1.0, Nt, dtype=np.float32)
XX, TT = np.meshgrid(x_coords, t_coords, indexing="ij")
coords = np.stack([XX.ravel(), TT.ravel()], axis=1).astype(np.float32)
coords_t = torch.tensor(coords, dtype=torch.float32, device=device)


# ============================================================
# Inference
# ============================================================
if use_manual_params:
    X_mifno = ((X_manual_mifno - mifno_scalers["x_mean"]) / mifno_scalers["x_std"]).astype(np.float32)
    X_mionet = ((X_manual_mionet - mionet_scalers["x_mean"]) / mionet_scalers["x_std"]).astype(np.float32)
else:
    X_mifno = ((X - mifno_scalers["x_mean"]) / mifno_scalers["x_std"]).astype(np.float32)
    X_mionet = ((X - mionet_scalers["x_mean"]) / mionet_scalers["x_std"]).astype(np.float32)

with torch.no_grad():
    _ = mifno_model(torch.tensor(X_mifno[:1], dtype=torch.float32, device=device))
    _ = mionet_model(torch.tensor(X_mionet[:1], dtype=torch.float32, device=device), coords_t)

sync(device)
t0 = time.time()
with torch.no_grad():
    pred_mifno = mifno_model(torch.tensor(X_mifno, dtype=torch.float32, device=device)).cpu().numpy()
sync(device)
mifno_time = time.time() - t0

sync(device)
t1 = time.time()
with torch.no_grad():
    pred_mionet = mionet_model(torch.tensor(X_mionet, dtype=torch.float32, device=device), coords_t).cpu().numpy()
sync(device)
mionet_time = time.time() - t1

Y_pred_temp_mifno = pred_mifno[:, 0] * mifno_scalers["yT_std"] + mifno_scalers["yT_mean"]
Y_pred_stress_mifno = pred_mifno[:, 1] * mifno_scalers["yS_std"] + mifno_scalers["yS_mean"]

Y_pred_temp_mionet = pred_mionet[:, 0] * mionet_scalers["yT_std"] + mionet_scalers["yT_mean"]
Y_pred_stress_mionet = pred_mionet[:, 1] * mionet_scalers["yS_std"] + mionet_scalers["yS_mean"]

#
# ============================================================
# Metrics
# ============================================================
if use_manual_params:
    temp_l2_mifno = rel_l2(Y_pred_temp_mifno[0], Y_temp[case_id])
    stress_l2_mifno = rel_l2(Y_pred_stress_mifno[0], Y_stress[case_id])
    temp_l2_mionet = rel_l2(Y_pred_temp_mionet[0], Y_temp[case_id])
    stress_l2_mionet = rel_l2(Y_pred_stress_mionet[0], Y_stress[case_id])
else:
    temp_l2_mifno = rel_l2(Y_pred_temp_mifno, Y_temp)
    stress_l2_mifno = rel_l2(Y_pred_stress_mifno, Y_stress)
    temp_l2_mionet = rel_l2(Y_pred_temp_mionet, Y_temp)
    stress_l2_mionet = rel_l2(Y_pred_stress_mionet, Y_stress)

fem_time_per_sample = float(timing_summary.get("solve_time_mean_s", 30.5))

mifno_time_per_sample = mifno_time / max(len(X_mifno), 1)
mionet_time_per_sample = mionet_time / max(len(X_mionet), 1)

mifno_speedup = fem_time_per_sample / max(mifno_time_per_sample, 1e-12)
mionet_speedup = fem_time_per_sample / max(mionet_time_per_sample, 1e-12)

top1, top2, top3, top4 = st.columns(4)
top1.metric("MIFNO Temp L2", f"{temp_l2_mifno*100:.3f}%")
top2.metric("MIFNO Stress L2", f"{stress_l2_mifno*100:.3f}%")
top3.metric("MIONet Temp L2", f"{temp_l2_mionet*100:.3f}%")
top4.metric("MIONet Stress L2", f"{stress_l2_mionet*100:.3f}%")

if timing_summary is not None:
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("FEM mean solve time", f"{timing_summary['solve_time_mean_s']:.3f} s")
    b2.metric("FEM median solve time", f"{timing_summary['solve_time_median_s']:.3f} s")
    b3.metric("MIFNO speed-up", f"{mifno_speedup:.1f}×")
    b4.metric("MIONet speed-up", f"{mionet_speedup:.1f}×")
    
# Case-local arrays for plotting
if use_manual_params:
    fem_temp_case = Y_temp[case_id]
    fem_stress_case = Y_stress[case_id]
    mifno_temp_case = Y_pred_temp_mifno[0]
    mifno_stress_case = Y_pred_stress_mifno[0]
    mionet_temp_case = Y_pred_temp_mionet[0]
    mionet_stress_case = Y_pred_stress_mionet[0]
else:
    fem_temp_case = Y_temp[case_id]
    fem_stress_case = Y_stress[case_id]
    mifno_temp_case = Y_pred_temp_mifno[case_id]
    mifno_stress_case = Y_pred_stress_mifno[case_id]
    mionet_temp_case = Y_pred_temp_mionet[case_id]
    mionet_stress_case = Y_pred_stress_mionet[case_id]


# ============================================================
# Zone KPI summary
# ============================================================
st.markdown("## Zone KPI summary")

if mode == "Dataset evaluation":
    kpi_source_temp = fem_temp_case
    kpi_source_stress = fem_stress_case
    st.caption("FEM-based KPI summary")
else:
    kpi_source_temp = mifno_temp_case
    kpi_source_stress = mifno_stress_case
    st.caption("MIFNO-based KPI summary")

kpi_rows = build_zone_kpi_table(
    kpi_source_temp,
    kpi_source_stress,
    t_axis,
    ZONE_TIME_WINDOWS,
    surface_idx=surface_idx,
    mid_idx=mid_idx,
)

st.dataframe(kpi_rows, use_container_width=True)
# ============================================================
# Plot rows
# ============================================================
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(8, 4))

    if mode == "Dataset evaluation":
        ax.plot(t_axis, fem_temp_case[surface_idx, :], "--", label="FEM")
        ax.plot(t_axis, mifno_temp_case[surface_idx, :], label="MIFNO")
        ax.plot(t_axis, mionet_temp_case[surface_idx, :], label="MIONet")
    else:
        ax.plot(t_axis, mifno_temp_case[surface_idx, :], label="MIFNO")
        ax.plot(t_axis, mionet_temp_case[surface_idx, :], label="MIONet")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Surface temperature vs time", fontsize=14, fontweight="semibold", pad=18)
    ax.grid(True)
    ax.legend()
    if show_zones:
        add_zone_background_time(ax, t_axis, ZONE_TIME_WINDOWS, ZONE_COLORS, alpha=zone_alpha)
    st.pyplot(fig)

with right:
    fig, ax = plt.subplots(figsize=(8, 4))

    if mode == "Dataset evaluation":
        ax.plot(x_axis, fem_temp_case[surface_idx, :], "--", label="FEM")
        ax.plot(x_axis, mifno_temp_case[surface_idx, :], label="MIFNO")
        ax.plot(x_axis, mionet_temp_case[surface_idx, :], label="MIONet")
    else:
        ax.plot(x_axis, mifno_temp_case[surface_idx, :], label="MIFNO")
        ax.plot(x_axis, mionet_temp_case[surface_idx, :], label="MIONet")

    ax.set_xlabel("Lehr distance x (m)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Surface temperature vs distance", fontsize=14, fontweight="semibold", pad=18)
    ax.grid(True)
    ax.legend()
    if show_zones:
        add_zone_background_distance(ax, x_axis, velocity, ZONE_TIME_WINDOWS, ZONE_COLORS, alpha=zone_alpha)
    st.pyplot(fig)

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(8, 4))

    if mode == "Dataset evaluation":
        ax.plot(t_axis, fem_stress_case[surface_idx, :], "--", label="FEM surface")
        ax.plot(t_axis, mifno_stress_case[surface_idx, :], label="MIFNO surface")
        ax.plot(t_axis, mionet_stress_case[surface_idx, :], label="MIONet surface")
        ax.plot(t_axis, fem_stress_case[mid_idx, :], "--", label="FEM mid")
        ax.plot(t_axis, mifno_stress_case[mid_idx, :], label="MIFNO mid")
        ax.plot(t_axis, mionet_stress_case[mid_idx, :], label="MIONet mid")
    else:
        ax.plot(t_axis, mifno_stress_case[surface_idx, :], label="MIFNO surface")
        ax.plot(t_axis, mionet_stress_case[surface_idx, :], label="MIONet surface")
        ax.plot(t_axis, mifno_stress_case[mid_idx, :], label="MIFNO mid")
        ax.plot(t_axis, mionet_stress_case[mid_idx, :], label="MIONet mid")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stress (Pa)")
    ax.set_title("Stress vs time", fontsize=14, fontweight="semibold", pad=18)
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)
    if show_zones:
        add_zone_background_time(ax, t_axis, ZONE_TIME_WINDOWS, ZONE_COLORS, alpha=zone_alpha)
    st.pyplot(fig)

with right:
    st.pyplot(plot_stress_vs_distance(
        x_axis,
        fem_stress_case,
        mifno_stress_case,
        mionet_stress_case,
        surface_idx,
        mid_idx,
        velocity,
        ZONE_TIME_WINDOWS,
        ZONE_COLORS,
        show_zones=show_zones,
        zone_alpha=zone_alpha,
        show_fem=(mode == "Dataset evaluation"),
    ))

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(8, 4))

    if mode == "Dataset evaluation":
        ax.plot(fem_temp_case[:, t_idx], "--", label="FEM")
        ax.plot(mifno_temp_case[:, t_idx], label="MIFNO")
        ax.plot(mionet_temp_case[:, t_idx], label="MIONet")
    else:
        ax.plot(mifno_temp_case[:, t_idx], label="MIFNO")
        ax.plot(mionet_temp_case[:, t_idx], label="MIONet")

    ax.set_xlabel("Thickness index")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"Through-thickness temperature at t_idx={t_idx}", fontsize=14, fontweight="semibold", pad=18)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

with right:
    st.pyplot(plot_through_thickness_stress_at_t(
        fem_stress_case,
        mifno_stress_case,
        mionet_stress_case,
        t_idx,
        show_fem=(mode == "Dataset evaluation"),
    ))

left, right = st.columns(2)

# ============================================================
# Temperature and stress accuracy
# ============================================================
with left:
    fig, ax = plt.subplots(figsize=(8, 4))

    values = [0.0, temp_l2_mifno * 100, temp_l2_mionet * 100]
    bars = ax.bar(["FEM", "MIFNO", "MIONet"], values)

    ax.set_title("Temperature accuracy", fontsize=14, fontweight="semibold", pad=18)
    ax.set_ylabel("Relative L2 Error (%)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ymax = max(values)
    ax.set_ylim(0, ymax * 1.3 if ymax > 0 else 1.0)

    for b, val in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width()/2,
            b.get_height() + 0.05 * ymax,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=False
        )

    st.pyplot(fig)
    
with right:
    fig, ax = plt.subplots(figsize=(8, 4))

    values = [0.0, stress_l2_mifno * 100, stress_l2_mionet * 100]
    bars = ax.bar(["FEM", "MIFNO", "MIONet"], values)

    ax.set_title("Stress accuracy", fontsize=14, fontweight="semibold", pad=18)
    ax.set_ylabel("Relative L2 Error (%)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ymax = max(values)
    ax.set_ylim(0, ymax * 1.3 if ymax > 0 else 1.0)

    for b, val in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width()/2,
            b.get_height() + 0.05 * ymax,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=False
        )

    st.pyplot(fig)

left, right = st.columns(2)
# ============================================================
# Computation time
# ============================================================
with left:
    fig, ax = plt.subplots(figsize=(8, 4))

    values = [fem_time_per_sample, mifno_time_per_sample, mionet_time_per_sample]
    bars = ax.bar(["FEM", "MIFNO", "MIONet"], values)

    ax.set_title("Computation time", fontsize=14, fontweight="semibold", pad=18)
    ax.set_ylabel("Time per sample (s)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ymax = max(values)
    ax.set_ylim(0, ymax * 1.25)

    for b, val in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width()/2,
            b.get_height() + 0.03 * ymax,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=False
        )

    st.pyplot(fig)


# ============================================================
# Speed-up (centered)
# ============================================================
#col1, col2, col3 = st.columns([1, 2, 1])

with right:
    fig, ax = plt.subplots(figsize=(8, 4))

    values = [1.0, mifno_speedup, mionet_speedup]
    bars = ax.bar(["FEM", "MIFNO", "MIONet"], values)

    ax.set_title("Speed-up vs FEM", fontsize=14, fontweight="semibold", pad=18)
    ax.set_ylabel("Speed-up (×)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ymax = max(values)
    ax.set_ylim(0, ymax * 1.15)

    for b, val in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02 * ymax,
            f"{val:.2f}×",
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=False
        )

    st.pyplot(fig)
      
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mifno_temp_case, cmap="RdYlBu_r", aspect="auto", origin="lower")
    ax.set_title("MIFNO temperature field", fontsize=14,
    fontweight="semibold",)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Thickness index")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

with right:
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mionet_temp_case, cmap="RdYlBu_r", aspect="auto", origin="lower")
    ax.set_title("MIONet temperature field", fontsize=14,
    fontweight="semibold",)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Thickness index")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

left, right = st.columns(2)

with left:
    st.pyplot(plot_field_map(
        mifno_stress_case,
        "MIFNO stress field",
        "Stress (Pa)", 
        cmap="PuOr"

    ))

with right:
    st.pyplot(plot_field_map(
        mionet_stress_case,
        "MIONet stress field",
        "Stress (Pa)",
        cmap="PuOr"
    ))

left, right = st.columns(2)

with left:
    st.pyplot(plot_error_map(
        mifno_temp_case,
        fem_temp_case,
        "MIFNO |Temperature Error|",
        "|Δ Temperature| (K)",
    ))

with right:
    st.pyplot(plot_error_map(
        mionet_temp_case,
        fem_temp_case,
        "MIONet |Temperature Error|",
        "|Δ Temperature| (K)",
    ))
    
left, right = st.columns(2)

with left:
    st.pyplot(plot_error_map(
        mifno_stress_case,
        fem_stress_case,
        "MIFNO |Stress Error|",
        "|Δ Stress| (Pa)",
    ))

with right:
    st.pyplot(plot_error_map(
        mionet_stress_case,
        fem_stress_case,
        "MIONet |Stress Error|",
        "|Δ Stress| (Pa)",
    ))

#next step to do better profissional repo

