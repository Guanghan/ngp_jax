import jax, flax, optax
from jax import lax
import flax.linen as nn

import numpy as np
import jax.numpy as jnp

from typing import Any, Callable
import functools

from hash_encoding import HashEmbedder

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


class ToyNeRF(nn.Module):
    """
    The simplest toy example of NeRF:
    (1) xyz and direction share the same encoding, default_freqs = 6
    (2) output (r, g, b, sigma) together, do NOT use the two-stage scheme: \
        first input xyz, output sigma
        then input dir and sigma, output rgb
    (3) do not use directions as condition to the MLPs
    """
    dtype: Any = jnp.float32
    precision: Any = lax.Precision.DEFAULT
    apply_positional_encoding: bool = config.apply_positional_encoding

    @nn.compact
    def __call__(self, input_points):
        # Apply positional encoding to raw input points
        x = positional_encoding(input_points) if self.apply_positional_encoding \
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
            x = jnp.concatenate([x, input_points], axis=-1) \
                if i==config.num_dense_layers//2 else x
        
        # output consists of 4 values: (r, g, b, sigma)
        x = nn.Dense(4, dtype=self.dtype, precision=self.precision)(x)
        return x


class SimpleNeRF(nn.Module):
    """
    An even simpler NeRF than the vanila NeRF:
    (1) xyz and direction share the same encoding, default_freqs = 6
    (2) output (r, g, b, sigma) together, do NOT use the two-stage scheme: \
        first input xyz, output sigma
        then input dir and sigma, output rgb
    (3) do not use directions as condition to the MLPs
    """
    dtype: Any = jnp.float32
    precision: Any = lax.Precision.DEFAULT
    apply_positional_encoding: bool = config.apply_positional_encoding

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
            )(encoded_points)

            # relu
            x = nn.relu(x)

            # skip connection
            x = jnp.concatenate([x, encoded_points], axis=-1) \
                if i==config.num_dense_layers//2 else x
        
        # output consists of 4 values: (r, g, b, sigma)
        x = nn.Dense(4, dtype=self.dtype, precision=self.precision)(x)
        return x


class VanillaNeRF(nn.Module):
    """
    The original NeRF described in the paper:
    https://arxiv.org/abs/2003.08934

    x:  input points in shape of (-1, num_pts, feature)
    condition: jnp.ndarray(float32), [batch, feature], if not None, this
        variable will be part of the input to the second part of the MLP
        concatenated with the output vector of the first part of the MLP. If
        None, only the first part of the MLP will be used with input x. In the
        original paper, this variable is the view direction.
    """

    @nn.compact
    def __call__(self, x, condition=None):
        #_, num_pts, feature_dim = x.shape
        #x = x.reshape([-1, feature_dim])
        num_pts, feature_dim = x.shape

        dense_layer = functools.partial(nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())

        inputs = x
        for i in range(config.num_dense_layers):
            # fc layer
            x = dense_layer(config.dense_layer_width)(x)
            x = nn.relu(x)

            if i % config.skip_layer == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis = -1)
        
        raw_sigma = dense_layer(1)(x).reshape([-1, num_pts, 1])

        if condition is not None:
            bottleneck = dense_layer(config.dense_layer_width)(x)

            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            condition = jnp.tile(condition[:, None, :], (1, num_pts, 1))

            # Collapse the [batch, num_samples, feature] tensor to
            # [batch * num_samples, feature] so that it can be fed into nn.Dense.
            condition = condition.reshape([-1, condition.shape[-1]])

            x = jnp.concatenate([bottleneck, condition], axis=-1)

            # use 1 extra layer to align with the original nerf model
            for i in range(config.num_dense_layers_dir):
                x = dense_layer(config.dense_layer_width_dir)(x)
                x = nn.relu(x)
        
        raw_rgb = dense_layer(3)(x).reshape([-1, num_pts, 3])
        return jnp.concatenate([raw_rgb, raw_sigma], axis=-1)


class HashNeRF(nn.Module):
    """
    The original NeRF but replaced with hash encoding

    x:  input points in shape of (-1, num_pts, feature)
    condition: jnp.ndarray(float32), [batch, feature], if not None, this
        variable will be part of the input to the second part of the MLP
        concatenated with the output vector of the first part of the MLP. If
        None, only the first part of the MLP will be used with input x. In the
        original paper, this variable is the view direction.
    """

    @nn.compact
    def __call__(self, inputs, condition=None):
        # Construct hash embedder
        embeder = HashEmbedder()
        
        # apply hash encoding 
        x = embeder(inputs)
        
        num_pts, feature_dim = x.shape

        dense_layer = functools.partial(nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())

        # point to the encoded input for future skip connection
        for i in range(config.num_dense_layers):
            # fc layer
            x = dense_layer(config.dense_layer_width)(x)
            x = nn.relu(x)

            if config.skip_layer != 0 and i % config.skip_layer == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis = -1)
        
        raw_sigma = dense_layer(1)(x).reshape([-1, num_pts, 1])

        if condition is not None:
            bottleneck = dense_layer(config.dense_layer_width)(x)

            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            condition = jnp.tile(condition[:, None, :], (1, num_pts, 1))

            # Collapse the [batch, num_samples, feature] tensor to
            # [batch * num_samples, feature] so that it can be fed into nn.Dense.
            condition = condition.reshape([-1, condition.shape[-1]])

            x = jnp.concatenate([bottleneck, condition], axis=-1)

            # use 1 extra layer to align with the original nerf model
            for i in range(config.num_dense_layers_dir):
                x = dense_layer(config.dense_layer_width_dir)(x)
                x = nn.relu(x)
        
        raw_rgb = dense_layer(3)(x).reshape([-1, num_pts, 3])
        return jnp.concatenate([raw_rgb, raw_sigma], axis=-1)


