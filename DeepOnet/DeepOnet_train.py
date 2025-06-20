import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

# --------------------
# 1. Load and preprocess data
# --------------------
a = np.loadtxt("temperature_over_time_0_50.txt").reshape(100, 49)
u = np.loadtxt("temperature_over_time_51_100.txt").reshape(100, 49)

# Normalize input and output data
a_min, a_max = a.min(), a.max()
u_min, u_max = u.min(), u.max()
a_norm = (a - a_min) / (a_max - a_min)
u_norm = (u - u_min) / (u_max - u_min)

# Generate spatial mesh (used for trunk input)
x = jnp.linspace(0, 1, 49)

# DeepONet input setup
X_branch = jnp.repeat(a_norm[:, None, :], 49, axis=1)  # (100, 49, 49)
X_trunk = jnp.tile(x[None, :, None], (100, 1, 1))  # (100, 49, 1)
Y = u_norm  # (100, 49)

# Flatten to 2D (N_samples * N_points, ...)
Xb = X_branch.reshape(-1, 49)
Xt = X_trunk.reshape(-1, 1)
Y_flat = Y.reshape(-1, 1)

# Train/test split
train_Xb, test_Xb = Xb[:80*49], Xb[80*49:]
train_Xt, test_Xt = Xt[:80*49], Xt[80*49:]
train_Y, test_Y = Y_flat[:80*49], Y_flat[80*49:]

# --------------------
# 2. Define DeepONet
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
        b_out = self.branch_net(branch_input)  # (..., p)
        t_out = self.trunk_net(trunk_input)    # (..., p)
        return jnp.sum(b_out * t_out, axis=-1, keepdims=True) + self.bias  # (..., 1)

# --------------------
# 3. Training setup
# --------------------
model = DeepONet(49, 1, 64, 64, key=jax.random.PRNGKey(0))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

def loss_fn(model, xb, xt, y):
    pred = jax.vmap(lambda b, t: model(b, t))(xb, xt)
    return jnp.mean((pred - y)**2)

@eqx.filter_jit
def make_step(model, opt_state, xb, xt, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, xb, xt, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

# --------------------
# 4. Training loop with timing
# --------------------
losses = []
start_time = time.time()
for epoch in range(3000):
    model, opt_state, loss = make_step(model, opt_state, train_Xb, train_Xt, train_Y)
    losses.append(loss)
    if epoch % 100 == 0:
        val_loss = loss_fn(model, test_Xb, test_Xt, test_Y)
        print(f"Epoch {epoch}, Train Loss: {loss:.6f}, Val Loss: {val_loss:.6f}")
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# Save model and normalization constants
with open("deeponet_model.eqx", "wb") as f:
    eqx.tree_serialise_leaves(f, model)

np.savez("normalization_constants.npz", a_min=a_min, a_max=a_max, u_min=u_min, u_max=u_max)

print("Model and normalization constants saved successfully.")
