# mlp_model_template.py
import jax
import jax.numpy as jnp
import equinox as eqx

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