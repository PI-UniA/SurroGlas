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

plt.rcParams["font.sans-serif"] = ["Arial"]
# ============================================================
# Lehr zones
# ============================================================
ZONE_TIME_WINDOWS = {
    "A1": (0.0,    54.7),
    "A2": (54.7,  109.6),
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
# Device helpers
# ============================================================
def pick_device(req: str) -> torch.device:
    req = req.lower().strip()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                return torch.device("cuda")
            except Exception:
                pass
        return torch.device("cpu")
    if req == "auto":
        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                return torch.device("cuda")
            except Exception:
                pass
    return torch.device("cpu")


def sync(device: torch.device):
    if device.type == "cuda":
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


# ============================================================
# General helpers
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


def build_param_vector_from_sidebar(param_names, sidebar_values):
    vals = []
    for name in param_names:
        if name not in sidebar_values:
            raise KeyError(f"Missing sidebar value for parameter '{name}'")
        vals.append(float(sidebar_values[name]))
    return np.array(vals, dtype=np.float32)


# ============================================================
# Zone background helpers
# ============================================================
def add_zone_background_time(ax, t_axis, zone_windows, zone_colors, alpha=0.35):
    t_min, t_max = t_axis[0], t_axis[-1]
    for zone, (t0, t1) in zone_windows.items():
        left  = max(t0, t_min)
        right = min(t1, t_max)
        if right > left:
            ax.axvspan(left, right, color=zone_colors.get(zone, "lightgray"),
                       alpha=alpha, zorder=0)
            center = 0.5 * (left + right)
            ax.text(center, 1.02, zone,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=11)
        if t_min <= t1 <= t_max:
            ax.axvline(t1, linestyle="--", linewidth=0.8, color="gray")


def add_zone_background_distance(ax, x_axis, velocity, zone_windows, zone_colors, alpha=0.35):
    x_min, x_max = x_axis[0], x_axis[-1]
    for zone, (t0, t1) in zone_windows.items():
        x0 = velocity * t0
        x1 = velocity * t1
        left  = max(x0, x_min)
        right = min(x1, x_max)
        if right > left:
            ax.axvspan(left, right, color=zone_colors.get(zone, "lightgray"),
                       alpha=alpha, zorder=0)
            center = 0.5 * (left + right)
            ax.text(center, 1.02, zone,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=11)
        if x_min <= x1 <= x_max:
            ax.axvline(x1, linestyle="--", linewidth=0.8, color="gray")


# ============================================================
# KPI table
# ============================================================
def build_zone_kpi_table(temp_field, stress_field, t_axis,
                          zone_windows, surface_idx=0, mid_idx=None):
    if mid_idx is None:
        mid_idx = temp_field.shape[0] // 2
    rows = []
    for zone, (t0, t1) in zone_windows.items():
        idx = np.where((t_axis >= t0) & (t_axis < t1))[0]
        if len(idx) == 0:
            continue
        temp_zone   = temp_field[:, idx]
        stress_zone = stress_field[:, idx]
        rows.append({
            "Zone":                    zone,
            "t_start (s)":             float(t0),
            "t_end (s)":               float(t1),
            "Mean surface T (K)":      float(np.mean(temp_zone[surface_idx])),
            "Exit surface T (K)":      float(temp_zone[surface_idx, -1]),
            "Mean mid T (K)":          float(np.mean(temp_zone[mid_idx])),
            "Max stress (MPa)":        float(np.max(stress_zone) / 1e6),
            "Min stress (MPa)":        float(np.min(stress_zone) / 1e6),
            "Exit surface stress (MPa)": float(stress_zone[surface_idx, -1] / 1e6),
            "Exit mid stress (MPa)":   float(stress_zone[mid_idx, -1] / 1e6),
        })
    return rows


# ============================================================
# Plot helpers
# ============================================================
def plot_stress_vs_distance(
    x_axis,
    fem_S,
    mifno_S,
    mionet_S,
    surface_idx,
    mid_idx,
    velocity,
    zone_windows,
    zone_colors,
    show_zones=True,
    zone_alpha=0.0,
    show_fem=True,
):

    fig, ax = plt.subplots(
        figsize=(6, 4),
        dpi=800
    )

    # ========================================================
    # FEM
    # ========================================================
    if show_fem:

        ax.plot(
            x_axis,
            fem_S[surface_idx] / 1e6,
            color="black",
            linestyle="-",
            linewidth=2.0,
            label="FEM surface"
        )

        ax.plot(
            x_axis,
            fem_S[mid_idx] / 1e6,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="FEM mid-plane"
        )

    # ========================================================
    # MIFNO
    # ========================================================

    ax.plot(
        x_axis,
        mifno_S[surface_idx] / 1e6,
        color="blue",
        linestyle="-",
        linewidth=2.0,
        label="MIFNO surface"
    )

    ax.plot(
        x_axis,
        mifno_S[mid_idx] / 1e6,
        color="blue",
        linestyle="--",
        linewidth=2.0,
        label="MIFNO mid-plane"
    )

    # ========================================================
    # MIONet
    # ========================================================

    ax.plot(
        x_axis,
        mionet_S[surface_idx] / 1e6,
        color="green",
        linestyle="-",
        linewidth=2.0,
        label="MIONet surface"
    )

    ax.plot(
        x_axis,
        mionet_S[mid_idx] / 1e6,
        color="green",
        linestyle="--",
        linewidth=2.0,
        label="MIONet mid-plane"
    )

    # ========================================================
    # Axis formatting
    # ========================================================

    ax.axhline(
        0,
        color="black",
        lw=0.7
    )

    ax.set_xlabel(
        "Lehr distance x (m)",
        fontsize=16
    )

    ax.set_ylabel(
        "Stress (MPa)",
        fontsize=16
    )

    ax.set_title(
        "Stress vs Lehr distance",
        fontsize=18,
        fontweight="semibold",
        pad=20
    )

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.grid(True)

    ax.legend(
        fontsize=11,
        ncol=2
    )

    # ========================================================
    # Zones
    # ========================================================

    if show_zones:

        add_zone_background_distance(
            ax,
            x_axis,
            velocity,
            zone_windows,
            zone_colors,
            zone_alpha
        )

    # ========================================================
    # Publication style
    # ========================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig


def plot_through_thickness_stress(fem_s, mifno_s, mionet_s,
                                   t_idx, z_axis, show_fem=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    if show_fem:
        ax.plot(z_axis, fem_s[:, t_idx]   / 1e6, "--", label="FEM")
    ax.plot(z_axis, mifno_s[:, t_idx]  / 1e6, label="MIFNO")
    ax.plot(z_axis, mionet_s[:, t_idx] / 1e6, label="MIONet")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title(f"Stress through thickness at t_idx={t_idx}",
                 fontsize=14, fontweight="semibold", pad=18)
    ax.grid(True); ax.legend()
    return fig


def plot_through_thickness_temp(fem_t, mifno_t, mionet_t,
                                 t_idx, z_axis, show_fem=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    if show_fem:
        ax.plot(z_axis, fem_t[:, t_idx], "--", label="FEM")
    ax.plot(z_axis, mifno_t[:, t_idx],  label="MIFNO")
    ax.plot(z_axis, mionet_t[:, t_idx], label="MIONet")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"Temperature through thickness at t_idx={t_idx}",
                 fontsize=14, fontweight="semibold", pad=18)
    ax.grid(True); ax.legend()
    return fig


def plot_field_map(field, title, cbar_label, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(field, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title, fontsize=13, fontweight="semibold")
    ax.set_xlabel("Time index"); ax.set_ylabel("Thickness index")
    fig.colorbar(im, ax=ax).set_label(cbar_label)
    return fig



def plot_field_map_vs_distance(field, x_axis, title, cbar_label, cmap="viridis",
                               velocity=None, zone_windows=None, zone_colors=None,
                               show_zones=True, zone_alpha=0.0,
                               scale_factor=1.0,
                               vmin=None, vmax=None):
    """
    Field map using the same logic as plot_case_annealing_multizone.py:
    x-axis: Lehr distance [m]
    y-axis: space index / thickness node [0 ... Nx-1]
    """

    nx_now = field.shape[0]

    fig, ax = plt.subplots(
        figsize=(6.0, 4.0),
        dpi=1200
    )

    im = ax.imshow(
        field / scale_factor,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin, vmax=vmax, 
        extent=[x_axis[0], x_axis[-1], 0, nx_now - 1],
        interpolation="nearest",
    )

    # ========================================================
    # Labels & title
    # ========================================================

    ax.set_xlabel(
        "Lehr distance x (m)",
        fontsize=18
    )

    ax.set_ylabel(
        "Thickness nodes",
        fontsize=18
    )

    ax.set_title(
        title,
        fontsize=18,
        fontweight="semibold",
        pad=20
    )

    # ========================================================
    # Tick font sizes
    # ========================================================

    ax.tick_params(
        axis="x",
        labelsize=15
    )

    ax.tick_params(
        axis="y",
        labelsize=15
    )

    # ========================================================
    # Colorbar
    # ========================================================

    cbar = fig.colorbar(im, ax=ax)

    cbar.set_label(
        cbar_label,
        fontsize=17
    )

    cbar.ax.tick_params(
        labelsize=14
    )

    # ========================================================
    # Zone background
    # ========================================================

    if show_zones and velocity is not None and zone_windows is not None:

        add_zone_background_distance(
            ax,
            x_axis,
            velocity,
            zone_windows,
            zone_colors or {},
            alpha=zone_alpha,
        )

    # ========================================================
    # Clean publication style
    # ========================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig

def plot_error_map(pred_field, true_field, title, cbar_label):
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(np.abs(pred_field - true_field),
                   aspect="auto", origin="lower")
    ax.set_title(title, fontsize=13, fontweight="semibold")
    ax.set_xlabel("Time index"); ax.set_ylabel("Thickness index")
    fig.colorbar(im, ax=ax).set_label(cbar_label)
    return fig


# ============================================================
# MIFNO model — from predict_MIFNO_multi-zone.py
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
# MIONet model — from predict_MIONET.py
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



# ============================================================
# Cached model + scaler loaders
# ============================================================
@st.cache_resource
def load_mifno(device_str: str):
    """Load MIFNO matching predict_MIFNO_multi-zone.py exactly."""
    device  = pick_device(device_str)
    sd      = "MIFNO"
    with open(os.path.join(sd, "latest_config.json")) as f:
        cfg = json.load(f)

    model = MIFNO(
        Nx          = cfg["Nx"],
        Nt          = cfg["Nt"],
        param_dim   = cfg["param_dim"],
        n_scales    = cfg["n_scales"],
        scale_width = cfg["scale_width"],
        width       = cfg["width"],
        depth       = cfg["depth"],
        modes_x     = cfg["modes_x"],
        modes_t     = cfg["modes_t"],
        dropout     = cfg.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(
        torch.load(os.path.join(sd, "latest_model_best.pt"),
                   map_location=device, weights_only=True),
        strict=True)
    model.eval()

    # Global scalar scalers (train_MIFNO saves single mean/std per field)
    scalers = {
        "x_mean":  np.load(os.path.join(sd, "x_scaler_mean_mifno_mz.npy")),
        "x_std":   np.load(os.path.join(sd, "x_scaler_scale_mifno_mz.npy")),
        "yT_mean": float(np.load(os.path.join(sd, "y_temp_mean_mz.npy")).flat[0]),
        "yT_std":  float(np.load(os.path.join(sd, "y_temp_std_mz.npy")).flat[0]),
        "yS_mean": float(np.load(os.path.join(sd, "y_stress_mean_mz.npy")).flat[0]),
        "yS_std":  float(np.load(os.path.join(sd, "y_stress_std_mz.npy")).flat[0]),
    }
    return cfg, model, scalers, device


@st.cache_resource
def load_mionet(device_str: str):
    """Load MIONet matching predict_MIONET.py exactly."""
    device  = pick_device(device_str)
    sd      = "MIONET"
    with open(os.path.join(sd, "latest_config_mionet.json")) as f:
        cfg = json.load(f)

    model = MIONet(
        param_dim    = cfg["param_dim"],
        latent_dim   = cfg["latent_dim"],
        branch_width = cfg["branch_width"],
        branch_depth = cfg["branch_depth"],
        trunk_width  = cfg["trunk_width"],
        trunk_depth  = cfg["trunk_depth"],
        Nx=cfg["Nx"], Nt=cfg["Nt"],
    ).to(device)

    model.load_state_dict(
        torch.load(os.path.join(sd, "latest_model_best_mionet.pt"),
                   map_location=device),
        strict=True)
    model.eval()

    # Per-pixel array scalers (train_MIONET uses StandardScaler flattened)
    scalers = {
        "x_mean":  np.load(os.path.join(sd, "x_scaler_mean_mionet_mz.npy")),
        "x_std":   np.load(os.path.join(sd, "x_scaler_scale_mionet_mz.npy")),
        "yT_mean": np.load(os.path.join(sd, "y_temp_scaler_mean_mionet_mz.npy")),
        "yT_std":  np.load(os.path.join(sd, "y_temp_scaler_scale_mionet_mz.npy")),
        "yS_mean": np.load(os.path.join(sd, "y_stress_scaler_mean_mionet_mz.npy")),
        "yS_std":  np.load(os.path.join(sd, "y_stress_scaler_scale_mionet_mz.npy")),
    }
    return cfg, model, scalers, device


@st.cache_data
def load_dataset(split: str, Nt: int, Nx: int):
    data_dir = os.path.join("results", split)
    pf = sorted([f for f in os.listdir(data_dir) if f.startswith("params_case_")],
                key=extract_case_idx)
    tf = sorted([f for f in os.listdir(data_dir) if f.startswith("temperature_all_case")],
                key=extract_case_idx)
    sf = sorted([f for f in os.listdir(data_dir) if f.startswith("stress_all_case")],
                key=extract_case_idx)
    X, Y_t, Y_s = [], [], []
    for p, t, s in zip(pf, tf, sf):
        X.append(np.loadtxt(os.path.join(data_dir, p), delimiter=",", skiprows=1))
        Y_t.append(np.loadtxt(os.path.join(data_dir, t)))
        Y_s.append(np.loadtxt(os.path.join(data_dir, s)))
    X   = np.array(X,   dtype=np.float32)
    Y_t = np.array(Y_t).reshape(len(X), Nt, Nx).transpose(0, 2, 1).astype(np.float32)
    Y_s = np.array(Y_s).reshape(len(X), Nt, Nx).transpose(0, 2, 1).astype(np.float32)
    return X, Y_t, Y_s


@st.cache_data
def load_fem_timing(split: str):
    p = os.path.join("results", split, "timing_summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def run_mifno_batched(model, X_sc, device, batch=4):
    """Batched MIFNO inference — avoids OOM for large N."""
    X_t = torch.tensor(X_sc, dtype=torch.float32)
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            xb = X_t[i : i + batch].to(device)
            chunks.append(model(xb).cpu())
            if device.type == "cuda":
                try: torch.cuda.empty_cache()
                except Exception: pass
    return torch.cat(chunks, dim=0).numpy()


# ============================================================
# Sidebar
# ============================================================
mode = st.sidebar.radio("Mode", ["Dataset evaluation", "Manual prediction"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Configuration")

device_choice = st.sidebar.selectbox("Device", ["cpu", "auto", "cuda"], index=0)
dt            = st.sidebar.number_input("dt (s)", value=0.1, step=0.01, format="%.3f")
velocity      = st.sidebar.number_input("Velocity (m/s)", value=0.16417, step=0.001, format="%.5f")
infer_batch   = st.sidebar.number_input("MIFNO batch size", value=4, min_value=1, max_value=64,
                                         help="Reduce to 1 or 2 if GPU runs out of memory")
show_zones    = st.sidebar.checkbox("Show Lehr zones", value=True)
zone_alpha    = st.sidebar.slider("Zone shading", 0.0, 0.8, 0.35, 0.05)

mifno_cfg,  mifno_model,  mifno_scalers,  mifno_device  = load_mifno(device_choice)
mionet_cfg, mionet_model, mionet_scalers, mionet_device = load_mionet(device_choice)

Nx = mifno_cfg["Nx"]
Nt = mifno_cfg["Nt"]
device = pick_device(device_choice)

# Physical axes — t starts at dt (first saved FEM timestep, not 0)
t_axis = (np.arange(Nt) + 1) * dt          # 0.1 ... 330.0
x_axis = velocity * t_axis
z_axis = np.linspace(-2.0, 2.0, Nx)        # mm, through thickness

surface_idx = 0
mid_idx     = Nx // 2

# MIONet coordinate grid
xc = np.linspace(0.0, 1.0, Nx, dtype=np.float32)
tc = np.linspace(0.0, 1.0, Nt, dtype=np.float32)
XX, TT   = np.meshgrid(xc, tc, indexing="ij")
coords   = np.stack([XX.ravel(), TT.ravel()], axis=1).astype(np.float32)
coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

# ============================================================
# Mode-specific sidebar inputs
# ============================================================
if mode == "Dataset evaluation":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset selection")
    split = st.sidebar.selectbox("Dataset split", ["test_unseen", "train"], index=0)
    X, Y_temp, Y_stress = load_dataset(split, Nt, Nx)
    timing_summary = load_fem_timing(split)
    case_id = st.sidebar.slider("Case ID", 0, len(X) - 1, 0)
    t_idx   = st.sidebar.slider("Time index", 0, Nt - 1, min(3299, Nt - 1))
    use_manual_params = False

else:
    split = "test_unseen"
    X, Y_temp, Y_stress = load_dataset(split, Nt, Nx)
    timing_summary = load_fem_timing(split)
    case_id = st.sidebar.slider("Reference FEM case", 0, len(X) - 1, 0)
    t_idx   = st.sidebar.slider("Time index", 0, Nt - 1, min(3299, Nt - 1))

    param_names = mifno_cfg.get("param_names", [])

    def param_default(name, fallback):
        if name in param_names:
            return float(X[case_id, param_names.index(name)])
        return float(fallback)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Zone-wise inputs")

    sidebar_vals = {
        "htc_A1":   st.sidebar.number_input("htc_A1",   value=param_default("htc_A1",   420.0), step=1.0),
        "htc_A2":   st.sidebar.number_input("htc_A2",   value=param_default("htc_A2",   420.0), step=1.0),
        "htc_B1":   st.sidebar.number_input("htc_B1",   value=param_default("htc_B1",   420.0), step=1.0),
        "htc_B2":   st.sidebar.number_input("htc_B2",   value=param_default("htc_B2",   420.0), step=1.0),
        "htc_C1":   st.sidebar.number_input("htc_C1",   value=param_default("htc_C1",   420.0), step=1.0),
        "T_amb_A1": st.sidebar.number_input("T_amb_A1", value=param_default("T_amb_A1", 873.0), step=1.0),
        "T_amb_A2": st.sidebar.number_input("T_amb_A2", value=param_default("T_amb_A2", 853.0), step=1.0),
        "T_amb_B1": st.sidebar.number_input("T_amb_B1", value=param_default("T_amb_B1", 827.0), step=1.0),
        "T_amb_B2": st.sidebar.number_input("T_amb_B2", value=param_default("T_amb_B2", 779.0), step=1.0),
        "T_amb_C1": st.sidebar.number_input("T_amb_C1", value=param_default("T_amb_C1", 757.0), step=1.0),
        "epsilon":  st.sidebar.number_input("epsilon",  value=param_default("epsilon",  0.85),  step=0.01, format="%.4f"),
        "sigma":    st.sidebar.number_input("sigma",    value=param_default("sigma",    5.67e-8), format="%.6e"),
    }

    mionet_param_names = mionet_cfg.get("param_names", param_names)
    X_manual_mifno  = build_param_vector_from_sidebar(param_names,        sidebar_vals).reshape(1, -1)
    X_manual_mionet = build_param_vector_from_sidebar(mionet_param_names, sidebar_vals).reshape(1, -1)
    use_manual_params = True


# ============================================================
# Inference
# ============================================================
if use_manual_params:
    X_sc_mifno  = ((X_manual_mifno  - mifno_scalers["x_mean"])  / mifno_scalers["x_std"]).astype(np.float32)
    X_sc_mionet = ((X_manual_mionet - mionet_scalers["x_mean"]) / mionet_scalers["x_std"]).astype(np.float32)
else:
    X_sc_mifno  = ((X - mifno_scalers["x_mean"])  / mifno_scalers["x_std"]).astype(np.float32)
    X_sc_mionet = ((X - mionet_scalers["x_mean"]) / mionet_scalers["x_std"]).astype(np.float32)

# MIFNO — batched to avoid OOM
sync(device)
t0 = time.time()
pred_mifno = run_mifno_batched(mifno_model, X_sc_mifno, device, batch=int(infer_batch))
sync(device)
mifno_time = time.time() - t0

# MIONet — full batch (much smaller memory footprint)
sync(device)
t1 = time.time()
with torch.no_grad():
    pred_mionet = mionet_model(
        torch.tensor(X_sc_mionet, device=device), coords_t).cpu().numpy()
sync(device)
mionet_time = time.time() - t1

# Inverse scale — MIFNO uses global scalars, MIONet uses per-pixel arrays
Y_pred_temp_mifno    = pred_mifno[:, 0]  * mifno_scalers["yT_std"]   + mifno_scalers["yT_mean"]
Y_pred_stress_mifno  = pred_mifno[:, 1]  * mifno_scalers["yS_std"]   + mifno_scalers["yS_mean"]
Y_pred_temp_mionet   = pred_mionet[:, 0] * mionet_scalers["yT_std"]  + mionet_scalers["yT_mean"]
Y_pred_stress_mionet = pred_mionet[:, 1] * mionet_scalers["yS_std"]  + mionet_scalers["yS_mean"]


# ============================================================
# Case-local arrays
# ============================================================
if use_manual_params:
    fem_T  = Y_temp[case_id];   fem_S  = Y_stress[case_id]
    mifno_T  = Y_pred_temp_mifno[0];   mifno_S  = Y_pred_stress_mifno[0]
    mionet_T = Y_pred_temp_mionet[0];  mionet_S = Y_pred_stress_mionet[0]
else:
    fem_T  = Y_temp[case_id];   fem_S  = Y_stress[case_id]
    mifno_T  = Y_pred_temp_mifno[case_id];   mifno_S  = Y_pred_stress_mifno[case_id]
    mionet_T = Y_pred_temp_mionet[case_id];  mionet_S = Y_pred_stress_mionet[case_id]


# ============================================================
# Metrics
# ============================================================
if use_manual_params:
    tl_mifno  = rel_l2(mifno_T,  fem_T)
    sl_mifno  = rel_l2(mifno_S,  fem_S)
    tl_mionet = rel_l2(mionet_T, fem_T)
    sl_mionet = rel_l2(mionet_S, fem_S)
else:
    tl_mifno  = rel_l2(Y_pred_temp_mifno,   Y_temp)
    sl_mifno  = rel_l2(Y_pred_stress_mifno, Y_stress)
    tl_mionet = rel_l2(Y_pred_temp_mionet,  Y_temp)
    sl_mionet = rel_l2(Y_pred_stress_mionet,Y_stress)

fem_t_per_sample    = float((timing_summary or {}).get("solve_time_mean_s", 30.5))
mifno_t_per_sample  = mifno_time  / max(len(X_sc_mifno),  1)
mionet_t_per_sample = mionet_time / max(len(X_sc_mionet), 1)
mifno_speedup  = fem_t_per_sample / max(mifno_t_per_sample,  1e-12)
mionet_speedup = fem_t_per_sample / max(mionet_t_per_sample, 1e-12)


# ============================================================
# Header metrics
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("MIFNO Temp L2",   f"{tl_mifno*100:.3f}%")
c2.metric("MIFNO Stress L2", f"{sl_mifno*100:.3f}%")
c3.metric("MIONet Temp L2",  f"{tl_mionet*100:.3f}%")
c4.metric("MIONet Stress L2",f"{sl_mionet*100:.3f}%")

if timing_summary:
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("FEM mean solve",   f"{timing_summary['solve_time_mean_s']:.1f} s")
    b2.metric("FEM median solve", f"{timing_summary['solve_time_median_s']:.1f} s")
    b3.metric("MIFNO speed-up",   f"{mifno_speedup:.0f}×")
    b4.metric("MIONet speed-up",  f"{mionet_speedup:.0f}×")


# ============================================================
# Zone KPI table
# ============================================================
st.markdown("## Zone KPI summary")
src_T = fem_T if mode == "Dataset evaluation" else mifno_T
src_S = fem_S if mode == "Dataset evaluation" else mifno_S
st.caption("FEM-based" if mode == "Dataset evaluation" else "MIFNO-based")
st.dataframe(
    build_zone_kpi_table(src_T, src_S, t_axis, ZONE_TIME_WINDOWS,
                         surface_idx, mid_idx),
    use_container_width=True)


# ============================================================
# Plots — row 1: temperature vs time / distance
# ============================================================
st.markdown("---")

left, right = st.columns(2)

# ============================================================
# Surface temperature vs time
# ============================================================
with left:

    fig, ax = plt.subplots(
        figsize=(6, 4),
        dpi=800
    )

    # ========================================================
    # FEM
    # ========================================================
    if mode == "Dataset evaluation":

        ax.plot(
            t_axis,
            fem_T[surface_idx],
            color="red",
            linestyle="-",
            linewidth=2.0,
            label="FEM surface"
        )

    # ========================================================
    # MIFNO
    # ========================================================

    ax.plot(
        t_axis,
        mifno_T[surface_idx],
        color="orange",
        linestyle="-",
        linewidth=2.0,
        label="MIFNO surface"
    )

    # ========================================================
    # MIONet
    # ========================================================

    ax.plot(
        t_axis,
        mionet_T[surface_idx],
        color="brown",
        linestyle="-",
        linewidth=2.0,
        label="MIONet surface"
    )

    # ========================================================
    # Formatting
    # ========================================================

    ax.set_xlabel(
        "Time (s)",
        fontsize=16
    )

    ax.set_ylabel(
        "Temperature (K)",
        fontsize=16
    )

    ax.set_title(
        "Surface temperature vs time",
        fontsize=18,
        fontweight="semibold",
        pad=20
    )

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.grid(True)

    ax.legend(
        fontsize=11
    )

    if show_zones:

        add_zone_background_time(
            ax,
            t_axis,
            ZONE_TIME_WINDOWS,
            ZONE_COLORS,
            zone_alpha
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# Surface temperature vs distance
# ============================================================
with right:

    fig, ax = plt.subplots(
        figsize=(6, 4),
        dpi=800
    )

    # ========================================================
    # FEM
    # ========================================================
    if mode == "Dataset evaluation":

        ax.plot(
            x_axis,
            fem_T[surface_idx],
            color="red",
            linestyle="-",
            linewidth=2.0,
            label="FEM surface"
        )

    # ========================================================
    # MIFNO
    # ========================================================

    ax.plot(
        x_axis,
        mifno_T[surface_idx],
        color="orange",
        linestyle="-",
        linewidth=2.0,
        label="MIFNO surface"
    )

    # ========================================================
    # MIONet
    # ========================================================

    ax.plot(
        x_axis,
        mionet_T[surface_idx],
        color="brown",
        linestyle="-",
        linewidth=2.0,
        label="MIONet surface"
    )

    # ========================================================
    # Formatting
    # ========================================================

    ax.set_xlabel(
        "Lehr distance x (m)",
        fontsize=16
    )

    ax.set_ylabel(
        "Temperature (K)",
        fontsize=16
    )

    ax.set_title(
        "Surface temperature vs distance",
        fontsize=18,
        fontweight="semibold",
        pad=20
    )

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.grid(True)

    ax.legend(
        fontsize=11
    )

    if show_zones:

        add_zone_background_distance(
            ax,
            x_axis,
            velocity,
            ZONE_TIME_WINDOWS,
            ZONE_COLORS,
            zone_alpha
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# Plots — row 2: stress vs time / distance  (MPa)
# ============================================================
left, right = st.columns(2)

with left:

    fig, ax = plt.subplots(figsize=(6, 4), dpi=800)

    # ========================================================
    # FEM
    # ========================================================
    if mode == "Dataset evaluation":

        # surface = solid
        ax.plot(
            t_axis,
            fem_S[surface_idx] / 1e6,
            color="black",
            linestyle="-",
            linewidth=2.0,
            label="FEM surface"
        )

        # mid-plane = dashed
        ax.plot(
            t_axis,
            fem_S[mid_idx] / 1e6,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="FEM mid-plane"
        )

    # ========================================================
    # MIFNO
    # ========================================================

    # surface = solid
    ax.plot(
        t_axis,
        mifno_S[surface_idx] / 1e6,
        color="blue",
        linestyle="-",
        linewidth=2.0,
        label="MIFNO surface"
    )

    # mid-plane = dashed
    ax.plot(
        t_axis,
        mifno_S[mid_idx] / 1e6,
        color="blue",
        linestyle="--",
        linewidth=2.0,
        label="MIFNO mid-plane"
    )

    # ========================================================
    # MIONet
    # ========================================================

    # surface = solid
    ax.plot(
        t_axis,
        mionet_S[surface_idx] / 1e6,
        color="green",
        linestyle="-",
        linewidth=2.0,
        label="MIONet surface"
    )

    # mid-plane = dashed
    ax.plot(
        t_axis,
        mionet_S[mid_idx] / 1e6,
        color="green",
        linestyle="--",
        linewidth=2.0,
        label="MIONet mid-plane"
    )

    # ========================================================
    # Axis formatting
    # ========================================================

    ax.axhline(
        0,
        color="black",
        lw=0.7
    )

    ax.set_xlabel(
        "Time (s)",
        fontsize=16
    )

    ax.set_ylabel(
        "Stress (MPa)",
        fontsize=16
    )

    ax.set_title(
        "Stress vs time",
        fontsize=18,
        fontweight="semibold",
        pad=20
    )

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.grid(True)

    ax.legend(
        ncol=2,
        fontsize=11
    )

    if show_zones:
        add_zone_background_time(
            ax,
            t_axis,
            ZONE_TIME_WINDOWS,
            ZONE_COLORS,
            zone_alpha
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

with right:

    fig = plot_stress_vs_distance(
        x_axis,
        fem_S,
        mifno_S,
        mionet_S,
        surface_idx,
        mid_idx,
        velocity,
        ZONE_TIME_WINDOWS,
        ZONE_COLORS,
        show_zones=show_zones,
        zone_alpha=zone_alpha,
        show_fem=(mode == "Dataset evaluation")
    )

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# Plots — row 3: through-thickness at t_idx
# ============================================================
left, right = st.columns(2)

with left:
    st.pyplot(plot_through_thickness_temp(
        fem_T, mifno_T, mionet_T, t_idx, z_axis,
        show_fem=(mode == "Dataset evaluation")))

with right:
    st.pyplot(plot_through_thickness_stress(
        fem_S, mifno_S, mionet_S, t_idx, z_axis,
        show_fem=(mode == "Dataset evaluation")))


# ============================================================
# Plots — row 4: accuracy & speed bar charts
# ============================================================

left, right = st.columns(2)

# ============================================================
# Temperature accuracy
# ============================================================
with left:

    fig, ax = plt.subplots(figsize=(6, 4), dpi=800)

    vals = [tl_mifno * 100, tl_mionet * 100]

    bars = ax.bar(
        ["MIFNO", "MIONet"],
        vals,
    )

    ax.set_title(
        "Temperature accuracy",
        fontsize=18,
        fontweight="semibold",
        pad=20,
    )

    ax.set_ylabel(
        "Relative L2 Error (%)",
        fontsize=16,
    )

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ymax = max(vals) or 1.0
    ax.set_ylim(0, ymax * 1.30)

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.05 * ymax,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
            clip_on=False,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# Stress accuracy
# ============================================================
with right:

    fig, ax = plt.subplots(figsize=(6, 4), dpi=800)

    vals = [sl_mifno * 100, sl_mionet * 100]

    bars = ax.bar(
        ["MIFNO", "MIONet"],
        vals,
    )

    ax.set_title(
        "Stress accuracy",
        fontsize=18,
        fontweight="semibold",
        pad=18,
    )

    ax.set_ylabel(
        "Relative L2 Error (%)",
        fontsize=16,
    )

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ymax = max(vals) or 1.0
    ax.set_ylim(0, ymax * 1.30)

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.05 * ymax,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
            clip_on=False,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# Second row
# ============================================================

left, right = st.columns(2)

# ============================================================
# Computation time
# ============================================================
with left:

    fig, ax = plt.subplots(figsize=(6, 4), dpi=800)

    vals = [
        fem_t_per_sample,
        mifno_t_per_sample,
        mionet_t_per_sample,
    ]

    bars = ax.bar(
        ["FEM", "MIFNO", "MIONet"],
        vals,
    )

    ax.set_title(
        "Computation time",
        fontsize=18,
        fontweight="semibold",
        pad=18,
    )

    ax.set_ylabel(
        "Time per sample (s)",
        fontsize=16,
    )

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ymax = max(vals) or 1.0
    ax.set_ylim(0, ymax * 1.25)

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.03 * ymax,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=13,
            clip_on=False,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# Speed-up vs FEM
# ============================================================
with right:

    fig, ax = plt.subplots(figsize=(6, 4), dpi=800)

    vals = [1.0, mifno_speedup, mionet_speedup]

    bars = ax.bar(
        ["FEM", "MIFNO", "MIONet"],
        vals,
    )

    ax.set_title(
        "Speed-up vs FEM",
        fontsize=18,
        fontweight="semibold",
        pad=18,
    )

    ax.set_ylabel(
        "Speed-up (×)",
        fontsize=16,
    )

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ymax = max(vals) or 1.0
    ax.set_ylim(0, ymax * 1.15)

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02 * ymax,
            f"{v:.1f}×",
            ha="center",
            va="bottom",
            fontsize=13,
            clip_on=False,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)
# ============================================================
# Plots — row 5: temperature field maps vs Lehr distance
# Same axis logic as plot_case_annealing_multizone.py:
# extent=[x_axis[0], x_axis[-1], 0, nx_now - 1]
# ============================================================
st.markdown("## Field maps vs Lehr distance")

# ── Shared temperature colour limits across FEM / MIFNO / MIONet ─────
_all_T  = np.concatenate([fem_T.ravel(), mifno_T.ravel(), mionet_T.ravel()])
t_vmin  = float(np.min(_all_T))
t_vmax  = float(np.max(_all_T))

cols = st.columns(3)

with cols[0]:
    fig = plot_field_map_vs_distance(
        fem_T,
        x_axis,
        "Temperature Field vs Lehr Distance",
        "Temperature (K)",
        cmap="RdYlBu_r",
        velocity=velocity,
        zone_windows=ZONE_TIME_WINDOWS,
        zone_colors=ZONE_COLORS,
        show_zones=show_zones,
        zone_alpha=0.0,
        vmin=t_vmin, vmax=t_vmax,
    )
    st.pyplot(fig); plt.close(fig)

with cols[1]:
    fig = plot_field_map_vs_distance(
        mifno_T,
        x_axis,
        "Temperature Field vs Lehr Distance",
        "Temperature (K)",
        cmap="RdYlBu_r",
        velocity=velocity,
        zone_windows=ZONE_TIME_WINDOWS,
        zone_colors=ZONE_COLORS,
        show_zones=show_zones,
        zone_alpha=0.0,
        vmin=t_vmin, vmax=t_vmax,
    )
    st.pyplot(fig); plt.close(fig)

with cols[2]:
    fig = plot_field_map_vs_distance(
        mionet_T,
        x_axis,
        "Temperature Field vs Lehr Distance",
        "Temperature (K)",
        cmap="RdYlBu_r",
        velocity=velocity,
        zone_windows=ZONE_TIME_WINDOWS,
        zone_colors=ZONE_COLORS,
        show_zones=show_zones,
        zone_alpha=0.0,
        vmin=t_vmin, vmax=t_vmax,
    )
    st.pyplot(fig); plt.close(fig)


# ============================================================
# Plots — row 6: stress field maps vs Lehr distance
# Stress scaling follows your plot_case_annealing_multizone.py: stress / 10e6
# ============================================================
cols = st.columns(3)

with cols[0]:
    fig = plot_field_map_vs_distance(
        fem_S,
        x_axis,
        "Stress Field vs Lehr Distance",
        "Stress (MPa)",
        cmap="PuOr",
        velocity=velocity,
        zone_windows=ZONE_TIME_WINDOWS,
        zone_colors=ZONE_COLORS,
        show_zones=show_zones,
        zone_alpha=0.0,
        scale_factor=10e6,
    )
    st.pyplot(fig); plt.close(fig)

with cols[1]:
    fig = plot_field_map_vs_distance(
        mifno_S,
        x_axis,
        "Stress Field vs Lehr Distance",
        "Stress (MPa)",
        cmap="PuOr",
        velocity=velocity,
        zone_windows=ZONE_TIME_WINDOWS,
        zone_colors=ZONE_COLORS,
        show_zones=show_zones,
        zone_alpha=0.0,
        scale_factor=10e6,
    )
    st.pyplot(fig); plt.close(fig)

with cols[2]:
    fig = plot_field_map_vs_distance(
        mionet_S,
        x_axis,
        "Stress Field vs Lehr Distance",
        "Stress (MPa)",
        cmap="PuOr",
        velocity=velocity,
        zone_windows=ZONE_TIME_WINDOWS,
        zone_colors=ZONE_COLORS,
        show_zones=show_zones,
        zone_alpha=0.0,
        scale_factor=10e6,
    )
    st.pyplot(fig); plt.close(fig)


# ============================================================
# Plots — row 7: original field maps by time index
# ============================================================
st.markdown("## Field maps by time index")

left, right = st.columns(2)
with left:
    st.pyplot(plot_field_map(mifno_T,  "MIFNO temperature field",  "Temperature (K)", "RdYlBu_r"))
with right:
    st.pyplot(plot_field_map(mionet_T, "MIONet temperature field", "Temperature (K)", "RdYlBu_r"))

left, right = st.columns(2)
with left:
    st.pyplot(plot_field_map(mifno_S  / 1e6, "MIFNO stress field",  "Stress (MPa)", "PuOr"))
with right:
    st.pyplot(plot_field_map(mionet_S / 1e6, "MIONet stress field", "Stress (MPa)", "PuOr"))


# ============================================================
# Plots — row 8: error maps
# ============================================================
st.markdown("## Error maps")

left, right = st.columns(2)
with left:
    st.pyplot(plot_error_map(mifno_T,  fem_T, "MIFNO |Temp Error|",   "|ΔT| (K)"))
with right:
    st.pyplot(plot_error_map(mionet_T, fem_T, "MIONet |Temp Error|",  "|ΔT| (K)"))

left, right = st.columns(2)
with left:
    st.pyplot(plot_error_map(mifno_S  / 1e6, fem_S / 1e6,
                              "MIFNO |Stress Error|",  "|Δσ| (MPa)"))
with right:
    st.pyplot(plot_error_map(mionet_S / 1e6, fem_S / 1e6,
                              "MIONet |Stress Error|", "|Δσ| (MPa)"))
