import jax, flax, optax
from jax import lax
import flax.linen as nn

import numpy as np
import jax.numpy as jnp

from typing import Any, Callable

from config import Config
config = Config()


def positional_encoding(inputs):
    batch_size, _ = inputs.shape
    
    # Apply vmap transform to vectorize the multiplication op
    inputs_freq = jax.vmap(
        lambda x: inputs * 2.0 ** x
    )(jnp.arange(config.positional_encoding_dims))

    periodic_fns = jnp.stack([jnp.sin(inputs_freq),
                              jnp.cos(inputs_freq)])
    periodic_fns = periodic_fns.swapaxes(0, 2).reshape([batch_size, -1])
    #periodic_fns = jnp.concatenate([inputs, periodic_fns], axis=-1)
    return periodic_fns


class BasicNeRF(nn.Module):
    dtype: Any = jnp.float32
    precision: Any = lax.Precision.DEFAULT
    apply_positional_concoding: bool = config.apply_positional_encoding

    @nn.compact
    def __call__(self, input_points):
        # Apply positional encoding to raw input points
        encoded_points = positional_encoding(input_points) if self.apply_positional_encoding \
            else input_points
        
        for i in range(config.num_dense_layers):
            # fc layer
            x = nn.Dense(config.dense_layer_width,
            dtype = self.dtype,
            precision = self.precision,
            kernel_init=jax.nn.initializers.glorot_uniform()
            )(x)

            # relu
            x = nn.relu(x)

            # skip connection
            x = jnp.concatenate([x, encoded_points], axis=-1) \
                if i==config.num_dense_layers//2 else x
        
        # output consists of 4 values: (r, g, b, sigma)
        x = nn.Dense(4, dtype=self.dtype, precision=self.precision)(x)
        return x