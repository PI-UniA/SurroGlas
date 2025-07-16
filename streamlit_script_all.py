import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests

st.title("Glass Cooling Temperature Prediction")

st.sidebar.header("Model Input Parameters")
model_name = st.sidebar.selectbox("Select Model", ["mlp", "mifno", "mionet"])

#model_name = st.sidebar.selectbox("Select Model", ["MLP", "MIFNO", "MIONet"])

htc = st.sidebar.slider("Heat Transfer Coefficient (htc)", 10.0, 300.0, 100.0)
epsilon = st.sidebar.slider("Emissivity (epsilon)", 0.5, 0.95, 0.85)
T_ambient = st.sidebar.slider("Ambient Temperature (K)", 270.0, 350.0, 273.15)
T_0 = st.sidebar.slider("Initial Temperature (K)", 800.0, 1200.0, 923.15)

selected_t = st.sidebar.slider("Select Time Index (0 - 199)", 0, 199, 190)

if st.button("Predict Temperature Profile"):
    with st.spinner("Sending request to backend and waiting for prediction..."):
        try:
            response = requests.post("http://localhost:8000/predict", json={
                "model": model_name,
                "htc": htc,
                "epsilon": epsilon,
                "T_ambient": T_ambient,
                "T_0": T_0
            })
            response.raise_for_status()
            data = response.json()
            temperature = np.array(data["prediction"])  # shape (49, 200)

            st.success("Prediction received!")
            st.subheader(f"Predicted Temperature over Space at t={selected_t} seconds")
            st.line_chart(temperature[:, selected_t])

            # Full heatmap
            st.subheader("Full Temperature Field (Space x Time)")
            fig, ax = plt.subplots(figsize=(10, 6))
            c = ax.imshow(temperature, aspect='auto', cmap='RdYlBu_r', origin='lower',
                         extent=[0, 199, 0, 48])
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Space Index")
            fig.colorbar(c, ax=ax, label="Temperature (K)")
            st.pyplot(fig)

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request failed: {e}")
