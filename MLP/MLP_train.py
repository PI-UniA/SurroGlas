#MLP_train.py 
import numpy as np
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler

# === Load parameter and temperature files ===
results_dir = "results"
param_files = sorted([f for f in os.listdir(results_dir) if f.startswith("params_case_")])
temp_files = sorted([f for f in os.listdir(results_dir) if f.startswith("temperature_all_case")])

X_params = []
Y_temperatures = []

for param_file, temp_file in zip(param_files, temp_files):
    params = np.loadtxt(os.path.join(results_dir, param_file), skiprows=1, delimiter=",")
    temps = np.loadtxt(os.path.join(results_dir, temp_file))

    if params.ndim == 0:
        params = np.expand_dims(params, axis=0)

    X_params.append(params)
    Y_temperatures.append(temps)

X_params = np.stack(X_params).astype(np.float32)  # (16, 4)
Y_temperatures = np.stack(Y_temperatures).astype(np.float32).reshape(16, 200, 49)  # (16, 200, 49)

# === Normalize ===
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X_params)

Y_flat = Y_temperatures.reshape(16, -1)
y_scaler = StandardScaler()
Y_scaled = y_scaler.fit_transform(Y_flat).reshape(16, 200, 49)

np.save("MLP/X_train_mlp.npy", X_scaled)
np.save("MLP/Y_train_mlp.npy", Y_scaled)

np.save("MLP/x_scaler_mean_mlp.npy", x_scaler.mean_)
np.save("MLP/x_scaler_scale_mlp.npy", x_scaler.scale_)
np.save("MLP/y_scaler_mean_mlp.npy", y_scaler.mean_)
np.save("MLP/y_scaler_scale_mlp.npy", y_scaler.scale_)

# === Flatten target ===
Y_train_flat = Y_scaled.reshape(16, -1)

# === Define MLP ===
class MLP(eqx.Module):
    layers: list

    def __init__(self, in_dim, out_dim, width, depth, *, key):
        keys = jax.random.split(key, depth + 1)
        self.layers = []
        self.layers.append(eqx.nn.Linear(in_dim, width, key=keys[0]))
        for i in range(1, depth):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i]))
        self.layers.append(eqx.nn.Linear(width, out_dim, key=keys[-1]))

    def __call__(self, x):
        def forward(xi):
            for layer in self.layers[:-1]:
                xi = jax.nn.relu(layer(xi))
            return self.layers[-1](xi)
        return jax.vmap(forward)(x)

in_dim = 4
out_dim = 200 * 49
width = 512
depth = 4
key = jax.random.PRNGKey(0)

model = MLP(in_dim, out_dim, width, depth, key=key)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(model)

def loss_fn(model, x, y):
    pred = model(x)
    return jnp.mean((pred - y) ** 2)

@eqx.filter_value_and_grad
def compute_loss(model, x, y):
    return loss_fn(model, x, y)

@eqx.filter_jit
def make_step(model, opt_state, x, y):
    loss, grads = compute_loss(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

X_train = jnp.array(X_scaled)
Y_train = jnp.array(Y_train_flat)

n_epochs = 10000
start_time = time.time()
for epoch in range(n_epochs):
    model, opt_state, loss = make_step(model, opt_state, X_train, Y_train)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")
end_time = time.time()
print(f"⏱️ Training time: {end_time - start_time:.2f} seconds")

eqx.tree_serialise_leaves("MLP/mlp_model.eqx", model)
