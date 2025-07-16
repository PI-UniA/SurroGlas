# MLP_predict.py
import jax
import jax.numpy as jnp
import equinox as eqx
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
        return jax.vmap(forward)(x)

# Load scalers
x_mean = np.load("MLP/x_scaler_mean_mlp.npy")
x_std = np.load("MLP/x_scaler_scale_mlp.npy")
y_mean = np.load("MLP/y_scaler_mean_mlp.npy")
y_std = np.load("MLP/y_scaler_scale_mlp.npy")

# Load model
in_dim = 4
out_dim = 200 * 49
width = 512
depth = 4
mlp_template = MLP(in_dim, out_dim, width, depth, key=jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves("MLP/mlp_model.eqx", like=mlp_template)

# Load and scale training data
X_train_raw = np.load("MLP/X_train_mlp.npy")
Y_train_scaled = np.load("MLP/Y_train_mlp.npy")

X_train_tensor = jnp.array(X_train_raw)
y_pred_flat = model(X_train_tensor)
y_pred_scaled = y_pred_flat.reshape(16, 200, 49)

# Inverse transform
Y_pred = np.array(y_pred_scaled).reshape(16, -1) * y_std + y_mean
Y_pred = Y_pred.reshape(16, 200, 49)
Y_true = Y_train_scaled.reshape(16, -1) * y_std + y_mean
Y_true = Y_true.reshape(16, 200, 49)

# Plot comparison
plt.plot(Y_true[0, 190, :], label='True (FEM)')
plt.plot(Y_pred[0, 190, :], label='Predicted (MLP)')
plt.title("Temperature Profile Comparison at t=190")
plt.xlabel("Space Index")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid()
plt.show()

# New input
new_input = np.array([[100, 0.85, 273.1, 923.15]], dtype=np.float32)
new_input_scaled = (new_input - x_mean) / x_std
new_input_tensor = jnp.array(new_input_scaled)

start = time.time()
new_pred_flat = model(new_input_tensor)
inference_time = time.time() - start

new_pred_scaled = new_pred_flat.reshape(1, 200, 49)
new_pred = np.array(new_pred_scaled).reshape(1, -1) * y_std + y_mean
new_pred = new_pred.reshape(1, 200, 49)

plt.plot(Y_true[1, 190, :], label='True Temperature Case 6')
plt.plot(new_pred[0, 190, :], label='Predicted Temperature for new input')
plt.title("New Input Prediction at t=190")
plt.xlabel("Spatial Index")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid()
plt.show()

print(f"⏱️ Inference time (new input): {inference_time:.6f} seconds")

def relative_l2_norm(pred, ref):
    diff_norm = np.linalg.norm(pred - ref)
    ref_norm = np.linalg.norm(ref)
    return diff_norm / ref_norm

rel_l2_set = [relative_l2_norm(Y_pred[i], Y_true[i]) for i in range(len(Y_pred))]
print("📈 Mean relative L2 error:", np.mean(rel_l2_set))
