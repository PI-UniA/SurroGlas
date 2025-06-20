import jax
import jax.numpy as jnp
import equinox as eqx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

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

# ------------------ Load Model ------------------
with open("mlp_model.pkl", "rb") as f:
    model = pickle.load(f)

# === Load data ===
X_train = np.load("X_train.npy")            # shape: (16, 4)
Y_train = np.load("Y_train.npy")      # shape: (16, 200, 49)

# === Predict on training data ===
y_pred_flat = model(X_train)      # shape: (16, 9800)
y_pred = y_pred_flat.reshape(16, 200, 49)

# ------------------ Inference Time: Train Set ------------------
start_time = time.time()
y_pred_flat = model(X_train)             # shape (16, 9800)
inference_time = time.time() - start_time

y_pred = y_pred_flat.reshape(16, 200, 49)

# === Plot prediction vs ground truth for case 0 and t = 170 ===
#plt.figure(figsize=(12, 8))
plt.plot(Y_train[0, 170, :], label='Y_temperatures (true/FEM)')
plt.plot(y_pred[0, 170, :], label='y_pred (MLP)')
plt.title("True vs Predicted Temperature Profiles over Space (case 0, t=17 seconds)")
plt.xlabel("Spatial Index")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid()
plt.show()

# ------------------ Predict for New Input ------------------
new_input = jnp.array([[100, 0.85, 273.1, 923.15]])  # Shape: (1, 4)

start_new = time.time()
new_pred_flat = model(new_input)
inference_new = time.time() - start_new

new_pred = new_pred_flat.reshape(1, 200, 49)

# === Compare prediction at t = 170 with known case 6 ===
#plt.figure(figsize=(12, 6))
plt.plot(Y_train[6, 170, :], label='Y_temperatures (case 6, t=10)', linestyle='--')
plt.plot(new_pred[0, 170, :], label='Predicted Temperature')
plt.title("Predicted Temperature Profile for New Input at t=10")
plt.xlabel("Spatial Index")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid()
plt.show()

# ------------------ Print Inference Times ------------------
print(f"⏱️  Inference time on all training data (batch of 16): {inference_time:.6f} seconds")
print(f"⏱️  Inference time for new input: {inference_new:.6f} seconds")

def relative_l2_norm(pred, ref):
    diff_norm = jnp.linalg.norm(pred - ref)
    ref_norm = jnp.linalg.norm(ref)
    return diff_norm / ref_norm

rel_l2_set = jax.vmap(relative_l2_norm)(y_pred, Y_train)
print("Mean relative L2 error:", jnp.mean(rel_l2_set))