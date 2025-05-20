# train_fno.py
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import time
from typing import Callable, List

# Load dataset
a = np.loadtxt("temperature_over_time_0_50.txt").reshape(100, 49)
u = np.loadtxt("temperature_over_time_51_100.txt").reshape(100, 49)


# Prepare dataset
a = a[:, jnp.newaxis, :]
u = u[:, jnp.newaxis, :]
#(250, 1, 49)

mesh = jnp.linspace(0, 1, u.shape[-1])

#(49,)
mesh_shape_corrected = jnp.repeat(mesh[jnp.newaxis, jnp.newaxis, :], u.shape[0], axis=0)

#(250, 1, 49)
a_with_mesh = jnp.concatenate((a, mesh_shape_corrected), axis=1)
#(250, 2, 49)
#So now each sample in the batch knows the corresponding spatial mesh.




train_x, test_x = a_with_mesh[:80], a_with_mesh[80:100]
train_y, test_y = u[:80], u[80:100]

#print(train_x.shape)
#print(train_y.shape)
#print(test_x.shape)
#print(test_y.shape)


# -- SpectralConv1d, FNOBlock1d, FNO1d definitions here --
# build FNO model
class SpectralConv1d(eqx.Module):
    real_weights: jax.Array
    imag_weights: jax.Array
    in_channels:int
    out_channels: int
    modes:int
    
    def __init__(self, in_channels, out_channels, modes,*,key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        # initalize real and imiginary components
        scale = 1.0/(in_channels*out_channels)
        real_key, imag_key = jax.random.split(key)
        self.real_weights = jax.random.uniform(real_key, (in_channels, out_channels, modes), minval=-scale, maxval=+scale,)
        self.imag_weights = jax.random.uniform(imag_key, (in_channels, out_channels, modes), minval=-scale, maxval=+scale,)
        
    def complex_multi1d(self, 
                        x_hat, w):
        return jnp.einsum("iM, ioM->oM", x_hat, w)
    #implement forward pass
    def __call__(self, x):
        #1. extract in_channels and spatial 
        channels, spatial_points = x.shape
        #shape of x_hat is (in_channels,spatials_points//2+1)
        x_hat = jnp.fft.rfft(x)
        x_hat_under_modes = x_hat[:,:self.modes]
        weights = self.real_weights +1j*self.imag_weights
        #shape of out_hat_under_modes is (outchannels, self.modes)
        out_hat_under_modes = self.complex_multi1d(x_hat_under_modes,weights)
        
        #2. shape of out_hat is (outchannels,spatials_points//2+1)
        out_hat = jnp.zeros(
            (self.out_channels, x_hat.shape[-1]),
            dtype=x_hat.dtype
        )
        # fill in zeros with number of modes
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
    

# define FNO model
class FNO1d(eqx.Module):
    lifting: eqx.nn.Conv1d
    fno_blocks: List[FNOBlock1d]
    projection: eqx.nn.Conv1d
    
    def __init__(self, in_channels, out_channels, modes, width, activation, n_blocks=4, *, key):
        key,lifting_key = jax.random.split(key)
        self.lifting = eqx.nn.Conv1d(in_channels,width,1,key=lifting_key,)
        self.fno_blocks = []
        for _ in range(n_blocks):
            key,sub_key = jax.random.split(key)
            self.fno_blocks.append(FNOBlock1d(width, width, modes, activation, key=sub_key))
            
        key, projection_key = jax.random.split(key)
        self.projection = eqx.nn.Conv1d(width, out_channels,1, key=projection_key,)
        
    def __call__(self, x):
        x = self.lifting(x)
        for block in self.fno_blocks:
            x = block(x)
        return self.projection(x)


# Create and JIT compile FNO model
fno = FNO1d(2, 1, 16, 64, jax.nn.relu, key=jax.random.PRNGKey(0))
fno = eqx.filter_jit(lambda x: x)(fno)

# Dataloader, loss function, optimizer
def dataloader(key, dataset_x, dataset_y, batch_size):
    n_samples = dataset_x.shape[0]
    n_batches = int(jnp.ceil(n_samples/batch_size))
    permutation = jax.random.permutation(key, n_samples)
    for batch_id in range(n_batches):
        start = batch_id * batch_size
        end = min((batch_id+1)*batch_size, n_samples)
        batch_indices = permutation[start:end]
        yield dataset_x[batch_indices], dataset_y[batch_indices]

def loss_fn(model, x, y):
    y_prediction = jax.vmap(model)(x)
    return jnp.mean(jnp.square(y_prediction - y))

optimizer = optax.adam(3e-4)
opt_state = optimizer.init(eqx.filter(fno, eqx.is_array))

@eqx.filter_jit
def make_step(model,state,x,y):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model,x,y)
    val_loss = loss_fn(model,test_x,test_y)
    updates, new_state = optimizer.update(grad,state,model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss, val_loss

# Train
loss_history = []
val_loss_history = []
shuffle_key = jax.random.PRNGKey(10)

start_time = time.time()

shuffle_key = jax.random.PRNGKey(10)

for epoch in range(400):
    epoch_train_loss = []
    epoch_val_loss = []

    shuffle_key, sub_key = jax.random.split(shuffle_key)
    for x_batch, y_batch in dataloader(sub_key, train_x, train_y, batch_size=20):
        fno, opt_state, loss, val_loss = make_step(fno, opt_state, x_batch, y_batch)
        loss_history.append(loss)
        val_loss_history.append(val_loss)
        epoch_train_loss.append(loss)
        epoch_val_loss.append(val_loss)
    
    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        avg_train = jnp.mean(jnp.array(epoch_train_loss))
        avg_val = jnp.mean(jnp.array(epoch_val_loss))
        print(f"Epoch {epoch+1}: Train Loss = {avg_train:.6f}, Validation Loss = {avg_val:.6f}")

        
import matplotlib.pyplot as plt
import numpy as np

# Assume loss_history and val_loss_history are available from training
# For safety, let's simulate short sample arrays if not defined
try:
    loss_np = np.array(loss_history)
    val_loss_np = np.array(val_loss_history)
except NameError:
    # Simulated example (to avoid errors if not defined)
    loss_np = np.exp(-0.1 * np.arange(1000)) + 0.01 * np.random.randn(1000)
    val_loss_np = np.exp(-0.1 * np.arange(1000)) + 0.02 * np.random.randn(1000)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(loss_np, label="Training Loss")
plt.plot(val_loss_np, label="Validation Loss", linestyle="--")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("FNO Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


end_time = time.time()
print(f"Training execution time: {end_time - start_time:.2f} seconds")

# Save model
eqx.tree_serialise_leaves("fno_model.eqx", fno)
print("Model saved successfully.")


 
