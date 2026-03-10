# app_streamlit.py
import os, io, json, time
import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

DEFAULT_API = os.environ.get("SURROGLAS_API", "http://localhost:8000")

st.set_page_config(
    page_title="SurroGlas – Thermal & Stress Prediction",
    page_icon="🔥",
    layout="wide",
)

# ── Sidebar: connection ──────────────────────────────────────
st.sidebar.header("🔌 Backend")
api_base = st.sidebar.text_input("FastAPI URL", DEFAULT_API)

@st.cache_data(ttl=10)
def get_health(api_base: str):
    try:
        r = requests.get(f"{api_base}/health", timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

health, health_err = get_health(api_base)

# ── Title ────────────────────────────────────────────────────
st.title("🔥 SurroGlas — Glass Cooling Prediction")
st.caption("Surrogate models for thermal & stress field prediction during glass cooling")

if health_err:
    st.error(f"Cannot reach backend: {health_err}")
    st.info("Start the server with:  `python backend_api_all.py`")
    st.stop()

Nx              = int(health.get("Nx", 49))
Nt              = int(health.get("Nt", 500))
available_models = health.get("models", ["mifno"])
backend_device  = health.get("device", "unknown")

# ── Backend status banner ─────────────────────────────────────
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Backend device", backend_device.upper())
col_s2.metric("Grid (Nx × Nt)", f"{Nx} × {Nt}")
col_s3.metric("Loaded models",  ", ".join(m.upper() for m in available_models))

# ── Architecture info (expander) ─────────────────────────────
with st.expander("ℹ️  Model architecture notes"):
    mifno_arch  = health.get("mifno_arch",  "—")
    mionet_arch = health.get("mionet_arch", "—")
    st.markdown(f"""
| Model | Architecture |
|---|---|
| **MIFNO** | {mifno_arch} |
| **MIONet** | {mionet_arch} |
| **MLP** | Dense MLP (JAX/Equinox), if loaded |

**MIFNO** encodes each of the 4 scalar inputs at 3 spatial scales independently,
fuses them via cross-scale attention, then processes with 6 FNO blocks where
the fused conditioning signal is injected at every layer.

**MIONet** uses 8 independent branch MLPs (4 inputs × 2 output channels),
fuses them via **element-wise product ⊙**, then dots with a shared trunk net
evaluated at every (x, t) grid coordinate.
    """)

st.markdown("---")

# ── Sidebar: model + parameters ──────────────────────────────
st.sidebar.header("🤖 Model")
model_name = st.sidebar.selectbox(
    "Select surrogate model",
    options=available_models,
    index=0,
    format_func=lambda m: {
        "mifno":  "MIFNO  (Multiscale Input FNO)",
        "mionet": "MIONet (Multi-Input Operator Net)",
        "mlp":    "MLP    (Dense baseline)",
    }.get(m, m.upper())
)

st.sidebar.header("⚙️  Input Parameters")
htc     = st.sidebar.slider("Heat Transfer Coeff. (htc)",  10.0,  300.0, 120.0, step=1.0,
                             help="Convective heat transfer coefficient [W/m²K]")
epsilon = st.sidebar.slider("Emissivity (ε)",              0.50,  0.95,  0.85,  step=0.01,
                             help="Surface emissivity (dimensionless, 0–1)")
sigma   = st.sidebar.slider("Stefan–Boltzmann (σ) ×10⁻⁸", 1.67,  10.67, 5.67,  step=0.01,
                             help="Effective radiation constant ×10⁻⁸") * 1e-8
alpha   = st.sidebar.slider("Thermal diffusivity (α)",     5.0,   20.0,  15.0,  step=0.5,
                             help="Thermal diffusivity [m²/s ×10⁻⁷]")

st.sidebar.header("🔍 View")
t_index = st.sidebar.slider("Time index (t_idx)", 0, max(0, Nt - 1), min(Nt - 1, 190), step=1,
                             help="Which time step to show in the spatial slice plots")

FIGSIZE  = (4.2, 2.8)
SAVE_DPI = 600


# ── Plot helpers ─────────────────────────────────────────────
def plot_field(field: np.ndarray, cmap: str, cbar_label: str, title: str = "",
               save_name: str | None = None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(field, aspect="auto", cmap=cmap, origin="lower",
                   extent=[0, field.shape[1] - 1, 0, field.shape[0] - 1])
    ax.set_xlabel("Time index"); ax.set_ylabel("Space index (x)")
    if title: ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax).set_label(cbar_label)
    fig.tight_layout()
    if save_name:
        fig.savefig(save_name, dpi=SAVE_DPI, bbox_inches="tight")
    return fig

def plot_slice(field: np.ndarray, t_idx: int, y_label: str, title: str = "",
               save_name: str | None = None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(field[:, t_idx], linewidth=1.8)
    ax.set_xlabel("Space index (x)"); ax.set_ylabel(y_label)
    if title: ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    if save_name:
        fig.savefig(save_name, dpi=SAVE_DPI, bbox_inches="tight")
    return fig

def download_npz(temp: np.ndarray, stress: np.ndarray, filename: str = "prediction_fields.npz"):
    buf = io.BytesIO()
    np.savez_compressed(buf, temperature=temp, stress=stress)
    st.download_button("⬇️ Download fields (.npz)", data=buf.getvalue(),
                       file_name=filename, mime="application/zip")


# ════════════════════════════════════════════════════════════
#  SINGLE PREDICTION
# ════════════════════════════════════════════════════════════
st.subheader("🔮 Single prediction")

left, right = st.columns([1, 2])
with left:
    st.markdown("**Input summary**")
    st.json({
        "model":   model_name,
        "htc":     htc,
        "epsilon": epsilon,
        "sigma":   float(f"{sigma:.4e}"),
        "alpha":   alpha,
    })

run = st.button("▶ Predict temperature & stress", type="primary")

if run:
    with st.spinner("Contacting backend…"):
        t0 = time.time()
        try:
            resp = requests.post(
                f"{api_base}/predict",
                json={"model": model_name, "htc": htc,
                      "epsilon": epsilon, "sigma": float(sigma), "alpha": alpha},
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()

        rt     = time.time() - t0
        api_ms = float(payload.get("inference_time_ms", 0.0))
        api_dev = payload.get("device", backend_device)

    st.success(f"✅ Round-trip {rt:.3f}s | model inference **{api_ms:.2f} ms** | device={api_dev}")

    temperature = np.array(payload["temperature"])   # (Nx, Nt)
    stress      = np.array(payload["stress"])         # (Nx, Nt)

    if temperature.shape != (Nx, Nt):
        st.error(f"Unexpected temperature shape {temperature.shape}, expected ({Nx},{Nt})")
        st.stop()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Grid", f"{Nx}×{Nt}")
    k2.metric("Inference", f"{api_ms:.2f} ms")
    k3.metric("Temp range (K)", f"{temperature.min():.1f} – {temperature.max():.1f}")
    k4.metric("Stress range (Pa)", f"{stress.min():.3g} – {stress.max():.3g}")

    # Spatial slices at t_index
    st.markdown(f"**Spatial profiles at t\_idx = {t_index}**")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_slice(temperature, t_index, "Temperature (K)",
                             title=f"{model_name.upper()} — Temp at t={t_index}",
                             save_name="temp_slice_600dpi.jpg"))
    with c2:
        st.pyplot(plot_slice(stress, t_index, "Stress (Pa)",
                             title=f"{model_name.upper()} — Stress at t={t_index}",
                             save_name="stress_slice_600dpi.jpg"))

    # Full (x,t) field maps
    st.markdown("**Full (x, t) field maps**")
    c3, c4 = st.columns(2)
    with c3:
        st.pyplot(plot_field(temperature, "RdYlBu_r", "Temperature (K)",
                             title=f"{model_name.upper()} — Temperature field",
                             save_name="temp_map_600dpi.jpg"))
    with c4:
        st.pyplot(plot_field(stress, "PuOr", "Stress (Pa)",
                             title=f"{model_name.upper()} — Stress field",
                             save_name="stress_map_600dpi.jpg"))

    download_npz(temperature, stress)


# ════════════════════════════════════════════════════════════
#  BATCH PREDICTION
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🚀 Batch prediction")
st.caption("Paste a JSON array of cases: `[[htc, epsilon, sigma, alpha], ...]`")

example_batch = [
    [120.0, 0.85, 5.67e-8, 15.0],
    [135.0, 0.69, 3.30e-8, 11.0],
    [90.0,  0.75, 4.20e-8, 12.5],
]
user_json = st.text_area("Cases JSON", value=json.dumps(example_batch, indent=2), height=160)

if st.button("▶ Run batch"):
    try:
        X = json.loads(user_json)
        if not isinstance(X, list) or any((not isinstance(r, list) or len(r) != 4) for r in X):
            raise ValueError("Expected list of [htc, ε, σ, α] rows.")
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    with st.spinner(f"Running batch of {len(X)} cases…"):
        try:
            r = requests.post(
                f"{api_base}/predict_batch",
                json={"model": model_name, "X": X},
                timeout=120,
            )
            r.raise_for_status()
            out = r.json()
        except requests.RequestException as e:
            st.error(f"Batch request failed: {e}")
            st.stop()

    B    = out.get("B", len(X))
    ms   = out.get("inference_time_ms", None)
    msps = out.get("inference_time_ms_per_sample", None)
    dev  = out.get("device", backend_device)

    msg = f"✅ {B} cases | device={dev}"
    if ms   is not None: msg += f" | total {ms:.2f} ms"
    if msps is not None: msg += f" ({msps:.3f} ms/sample)"
    st.success(msg)

    # Preview first case
    temp0   = np.array(out["temperature"][0])    # (Nx, Nt)
    stress0 = np.array(out["stress"][0])

    st.markdown("**Preview — case 0**")
    b1, b2 = st.columns(2)
    with b1:
        st.pyplot(plot_field(temp0,   "RdYlBu_r", "Temperature (K)",
                             title="Case 0 — Temperature",
                             save_name="batch_temp0_600dpi.jpg"))
    with b2:
        st.pyplot(plot_field(stress0, "PuOr",     "Stress (Pa)",
                             title="Case 0 — Stress",
                             save_name="batch_stress0_600dpi.jpg"))

    # Download full batch
    buf = io.BytesIO()
    np.savez_compressed(buf,
                        temperature=np.array(out["temperature"]),
                        stress=np.array(out["stress"]),
                        X=np.array(X))
    st.download_button("⬇️ Download batch (.npz)", buf.getvalue(),
                       "batch_predictions.npz", mime="application/zip")

    # Per-case summary table
    st.markdown("**Batch summary**")
    rows = []
    for i, (row, t_arr, s_arr) in enumerate(
        zip(X, out["temperature"], out["stress"])
    ):
        rows.append({
            "case": i,
            "htc": row[0], "ε": row[1], "σ": row[2], "α": row[3],
            "T_min": float(np.min(t_arr)), "T_max": float(np.max(t_arr)),
            "σ_min": float(np.min(s_arr)), "σ_max": float(np.max(s_arr)),
        })
    st.dataframe(rows, use_container_width=True)
