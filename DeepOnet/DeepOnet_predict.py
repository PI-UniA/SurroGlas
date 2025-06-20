import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import time

# --------------------
# Load data and normalization constants
# --------------------
a = np.loadtxt("temperature_over_time_0_50.txt").reshape(100, 49)
u = np.loadtxt("temperature_over_time_51_100.txt").reshape(100, 49)

norms = np.load("normalization_constants.npz")
a_min, a_max = norms["a_min"], norms["a_max"]
u_min, u_max = norms["u_min"], norms["u_max"]

a_norm = (a - a_min) / (a_max - a_min)
u_norm = (u - u_min) / (u_max - u_min)

x = jnp.linspace(0, 1, 49)
X_branch = jnp.repeat(a_norm[:, None, :], 49, axis=1)
X_trunk = jnp.tile(x[None, :, None], (100, 1, 1))
Y = u_norm

Xb = X_branch.reshape(-1, 49)
Xt = X_trunk.reshape(-1, 1)
Y_flat = Y.reshape(-1, 1)

# Test set only
test_Xb = Xb[80*49:]
test_Xt = Xt[80*49:]
test_Y = Y_flat[80*49:]

# --------------------
# Model Definition (must match training)
# --------------------
class MLP(eqx.Module):
    layers: list
    def __init__(self, sizes, key):
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [eqx.nn.Linear(sizes[i], sizes[i+1], key=k) for i, k in enumerate(keys)]
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x)

class DeepONet(eqx.Module):
    branch_net: MLP
    trunk_net: MLP
    bias: jax.Array
    def __init__(self, input_dim_branch, input_dim_trunk, hidden_dim, output_dim, *, key):
        branch_key, trunk_key, bias_key = jax.random.split(key, 3)
        self.branch_net = MLP([input_dim_branch, hidden_dim, hidden_dim, output_dim], key=branch_key)
        self.trunk_net = MLP([input_dim_trunk, hidden_dim, hidden_dim, output_dim], key=trunk_key)
        self.bias = jax.random.normal(bias_key, ())
    def __call__(self, branch_input, trunk_input):
        b_out = self.branch_net(branch_input)
        t_out = self.trunk_net(trunk_input)
        return jnp.sum(b_out * t_out, axis=-1, keepdims=True) + self.bias

# --------------------
# Load trained model
# --------------------
model = DeepONet(49, 1, 64, 64, key=jax.random.PRNGKey(0))
with open("deeponet_model.eqx", "rb") as f:
    model = eqx.tree_deserialise_leaves(f, model)

# --------------------
# Inference
# --------------------
start_infer = time.time()
pred_test = jax.vmap(lambda b, t: model(b, t))(test_Xb, test_Xt)
end_infer = time.time()
print(f"Inference time: {end_infer - start_infer:.4f} seconds")

# De-normalize
test_Y_denorm = test_Y * (u_max - u_min) + u_min
pred_test_denorm = pred_test * (u_max - u_min) + u_min

# Evaluation
rmse = jnp.sqrt(jnp.mean((pred_test_denorm - test_Y_denorm) ** 2))
print(f"Test RMSE: {rmse:.4f}")

# Plot one sample
idx = 0
start = idx * 49
end = (idx + 1) * 49

plt.plot(test_Xt[start:end].squeeze(), test_Y_denorm[start:end], label="True")
plt.plot(test_Xt[start:end].squeeze(), pred_test_denorm[start:end].squeeze(), label="DeepONet")
plt.xlabel("Spatial Coordinate")
plt.ylabel("Temperature")
plt.title("DeepONet Prediction vs True")
plt.legend()
plt.grid()
plt.show()

def relative_l2_norm(pred, ref):
    diff_norm = jnp.linalg.norm(pred - ref)
    ref_norm = jnp.linalg.norm(ref)
    return diff_norm / ref_norm

rel_l2_set = jax.vmap(relative_l2_norm)(pred_test_denorm, test_Y_denorm)
print("Mean relative L2 error:", jnp.mean(rel_l2_set))
