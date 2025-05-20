import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import time

# --------------------
# 1. Load dataset (same as during training)
# --------------------
a = np.loadtxt("temperature_over_time_0_50.txt").reshape(100, 49)
u = np.loadtxt("temperature_over_time_51_100.txt").reshape(100, 49)

a = a[:, jnp.newaxis, :]  # Shape: (250, 1, 49)
u = u[:, jnp.newaxis, :]  # Shape: (250, 1, 49)

mesh = jnp.linspace(0, 1, u.shape[-1])
mesh_shape_corrected = jnp.repeat(mesh[jnp.newaxis, jnp.newaxis, :], u.shape[0], axis=0)
a_with_mesh = jnp.concatenate((a, mesh_shape_corrected), axis=1)

# Only use the test set
train_x, test_x = a_with_mesh[:80], a_with_mesh[80:100]
train_y, test_y = u[:80], u[80:100]

# --------------------
# 2. Define the FNO model structure (same as during training)
# --------------------

from typing import Callable, List

class SpectralConv1d(eqx.Module):
    real_weights: jax.Array
    imag_weights: jax.Array
    in_channels: int
    out_channels: int
    modes: int

    def __init__(self, in_channels, out_channels, modes, *, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        real_key, imag_key = jax.random.split(key)
        self.real_weights = jax.random.uniform(real_key, (in_channels, out_channels, modes), minval=-scale, maxval=+scale)
        self.imag_weights = jax.random.uniform(imag_key, (in_channels, out_channels, modes), minval=-scale, maxval=+scale)

    def complex_multi1d(self, x_hat, w):
        return jnp.einsum("iM, ioM->oM", x_hat, w)

    def __call__(self, x):
        channels, spatial_points = x.shape
        x_hat = jnp.fft.rfft(x)
        x_hat_under_modes = x_hat[:, :self.modes]
        weights = self.real_weights + 1j * self.imag_weights
        out_hat_under_modes = self.complex_multi1d(x_hat_under_modes, weights)

        out_hat = jnp.zeros((self.out_channels, x_hat.shape[-1]), dtype=x_hat.dtype)
        out_hat = out_hat.at[:, :self.modes].set(out_hat_under_modes)
        out = jnp.fft.irfft(out_hat, n=spatial_points)
        return out

class FNOBlock1d(eqx.Module):
    spectral_conv: SpectralConv1d
    bypass_conv: eqx.nn.Conv1d
    activation: Callable

    def __init__(self, in_channels, out_channels, modes, activation, *, key):
        spectral_conv_key, bypass_conv_key = jax.random.split(key)
        self.spectral_conv = SpectralConv1d(in_channels, out_channels, modes, key=spectral_conv_key)
        self.bypass_conv = eqx.nn.Conv1d(in_channels, out_channels, kernel_size=1, key=bypass_conv_key)
        self.activation = activation

    def __call__(self, x):
        return self.activation(self.spectral_conv(x) + self.bypass_conv(x))

class FNO1d(eqx.Module):
    lifting: eqx.nn.Conv1d
    fno_blocks: List[FNOBlock1d]
    projection: eqx.nn.Conv1d

    def __init__(self, in_channels, out_channels, modes, width, activation, n_blocks=4, *, key):
        key, lifting_key = jax.random.split(key)
        self.lifting = eqx.nn.Conv1d(in_channels, width, 1, key=lifting_key)
        self.fno_blocks = []
        for _ in range(n_blocks):
            key, sub_key = jax.random.split(key)
            self.fno_blocks.append(FNOBlock1d(width, width, modes, activation, key=sub_key))

        key, projection_key = jax.random.split(key)
        self.projection = eqx.nn.Conv1d(width, out_channels, 1, key=projection_key)

    def __call__(self, x):
        x = self.lifting(x)
        for block in self.fno_blocks:
            x = block(x)
        return self.projection(x)

# --------------------
# 3. Load trained model
# --------------------

# Create a dummy model to match saved architecture
fno_dummy = FNO1d(2, 1, 16, 64, jax.nn.relu, key=jax.random.PRNGKey(0))
fno = eqx.tree_deserialise_leaves("fno_model.eqx", fno_dummy)

# --------------------
# 4. Prediction (Inference)
# --------------------

batched_forward = jax.jit(jax.vmap(fno))

start = time.time()
test_prediction = batched_forward(test_x)
inference_time = time.time() - start
print(f"Inference time: {inference_time:.6f} seconds")

# --------------------
# 5. Evaluation
# --------------------

def relative_l2_norm(pred, ref):
    diff_norm = jnp.linalg.norm(pred - ref)
    ref_norm = jnp.linalg.norm(ref)
    return diff_norm / ref_norm

rel_l2_set = jax.vmap(relative_l2_norm)(test_prediction, test_y)
print("Mean relative L2 error:", jnp.mean(rel_l2_set))


# --------------------
# 6. Plot a few results
# --------------------
#inputs of test model at t=0
plt.plot(test_x[0, 0], label="Initial condition")

plt.plot(test_y[0, 0], label="True at t=-")

plt.plot(test_prediction[0, 0], label="FNO prediction at t=-")
plt.legend()
plt.grid()
plt.show()

rmse = np.sqrt(np.mean((test_y - test_prediction) ** 2))
print(f"RMSE between FEM and FNO mean temperatures: {rmse:.3f} K")