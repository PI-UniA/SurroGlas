from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import equinox as eqx
import jax
import jax.numpy as jnp

from MIFNO.mifno_model import SimpleMIFNO
from MIONET.mionet_model import MIONet
from MLP.mlp_model_template import MLP

app = FastAPI()

Nx, Nt = 49, 200

# Load scalers
scalers = {}
for model_name in ["mlp", "mifno", "mionet"]:
    capital_name = model_name.upper() if model_name == "mlp" else model_name.capitalize()
    scalers[model_name] = {
        "x_mean": np.load(f"{capital_name}/x_scaler_mean_{capital_name}.npy"),
        "x_std": np.load(f"{capital_name}/x_scaler_scale_{capital_name}.npy"),
        "y_mean": np.load(f"{capital_name}/y_scaler_mean_{capital_name}.npy"),
        "y_std": np.load(f"{capital_name}/y_scaler_scale_{capital_name}.npy")
    }

# Load models
mifno_model = SimpleMIFNO(Nx, Nt)
mifno_model.load_state_dict(torch.load("MIFNO/mifno_model.pt", map_location=torch.device('cpu')))
mifno_model.eval()

mionet_model = MIONet(4, Nx, Nt)
mionet_model.load_state_dict(torch.load("MIONet/mionet_model.pt", map_location=torch.device('cpu')))
mionet_model.eval()

mlp_template = MLP(in_dim=4, out_dim=Nt * Nx, width=512, depth=4, key=jax.random.PRNGKey(0))
mlp_model = eqx.tree_deserialise_leaves("MLP/mlp_model.eqx", mlp_template)

models = {
    "mifno": mifno_model,
    "mionet": mionet_model,
    "mlp": mlp_model
}

class PredictRequest(BaseModel):
    model: str
    htc: float
    epsilon: float
    T_ambient: float
    T_0: float

@app.post("/predict")
def predict_temperature(req: PredictRequest):
    model_name = req.model.lower()
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model not supported")

    # Scale input
    x = np.array([[req.htc, req.epsilon, req.T_ambient, req.T_0]], dtype=np.float32)
    x_scaled = (x - scalers[model_name]["x_mean"]) / scalers[model_name]["x_std"]

    if model_name == "mlp":
        x_jax = jnp.array(x_scaled)
        pred_flat = models["mlp"](x_jax).reshape(Nt, Nx)
        pred = np.array(pred_flat)
    else:
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        with torch.no_grad():
            output = models[model_name](x_tensor)
        pred = output.squeeze(0).numpy().reshape(Nx, Nt)

    # Inverse transform
    pred_reshaped = pred.reshape(-1)
    pred_rescaled = pred_reshaped * scalers[model_name]["y_std"] + scalers[model_name]["y_mean"]
    pred_final = pred_rescaled.reshape(Nx, Nt)

    return {"prediction": pred_final.tolist()}
