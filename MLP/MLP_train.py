import numpy as np
import os
import pandas as pd
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

# Reload everything due to session reset
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

X_params = np.stack(X_params)  # shape: (16, 4)
Y_temperatures = np.stack(Y_temperatures)  # shape: (16, 9800)

# Reshape Y to (16, 200, 49) --> time, space
#
Y_temperatures = Y_temperatures.reshape(16, 200, 49)

# Generate time and space grids
time_grid = np.linspace(0, 1, 200)  # normalized time
space_grid = np.linspace(0, 1, 49)  # normalized space

# Create mesh grids and expand dimensions to match data
T, X = np.meshgrid(time_grid, space_grid, indexing='ij')  # shape: (200, 49)

# Expand to shape: (16, 1, 200, 49)
T_grid = np.tile(T[None, None, :, :], (16, 1, 1, 1))
X_grid = np.tile(X[None, None, :, :], (16, 1, 1, 1))

# Repeat parameters across the spatial-temporal grid
P_grids = []
for i in range(4):
    p = X_params[:, i][:, None, None, None]
    p_grid = np.tile(p, (1, 1, 200, 49))
    P_grids.append(p_grid)

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt

# Dummy data placeholders (replace with your actual data)
# X_train: (16, 4), Y_train: (16, 200, 49)
X_train = jnp.array(X_params)               # Shape: (16, 4)
Y_train = jnp.array(Y_temperatures)         # Shape: (16, 200, 49)

# Save training data
np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)

# Flatten Y_train targets to match MLP output
Y_train_flat = Y_train.reshape(Y_train.shape[0], -1)  # Shape: (16, 9800)


# Define MLP model
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
        return jax.vmap(forward)(x)  # x: (batch, 4)

# Hyperparameters
in_dim = 4
out_dim = 200 * 49
width = 512
depth = 4
key = jax.random.PRNGKey(0)

# Instantiate model
model = MLP(in_dim, out_dim, width, depth, key=key)

# Define optimizer and loss
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(model)

# Loss function
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

# Training loop
n_epochs = 10000
start_time = time.time()
for epoch in range(n_epochs):
    model, opt_state, loss = make_step(model, opt_state, X_train, Y_train_flat)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")
end_time = time.time()
print(f"⏱️ Training time: {end_time - start_time:.2f} seconds")

# Save model
with open("mlp_model.pkl", "wb") as f:
    pickle.dump(model, f)