from tkinter import N
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


def positional_encoding_trig(inputs):
    # Instead of computing [sin(x), cos(x)], we use the trig identity:
    # cos(x) = sin(x + pi/2), and perform a vectorized call to sin([x, x+pi/2])
    # https://www2.clarku.edu/faculty/djoyce/trig/identities.html
    min_degree, max_degree = config.positional_encoding_min_degree, \
                             config.positional_encoding_max_degree
    if min_degree == max_degree:
        return inputs
    
    scales = jnp.array([2**i for i in range(min_degree, max_degree)])
    xb = jnp.reshape((inputs[Ellipsis, None, :] * scales[:, None]),
                     list(inputs.shape[:-1]) + [-1]
    )
    four_feat = jnp.sin(jnp.concatenate([xb, xb+0.5*jnp.pi], axis=-1))
    return jnp.concatenate([inputs] + [four_feat], axis=-1)


class BasicNeRF(nn.Module):
    """
    An even simpler NeRF than the vanila NeRF:
    (1) xyz and direction share the same encoding, default_freqs = 6
    (2) output (r, g, b, sigma) together, do NOT use the two-stage scheme: \
        first input xyz, output sigma
        then input dir and sigma, output rgb
    """
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
    
class VanilaNeRF(nn.Module):
    """
    The original NeRF described in the paper:
    https://arxiv.org/abs/2003.08934
    """
    pass