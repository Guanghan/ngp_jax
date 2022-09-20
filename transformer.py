import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
#import SelfAttention, MultiHeadDotProductAttention, Module

from typing import Optional

class SelfAttention(nn.MultiHeadDotProductAttention):
    '''
    Self-attention with a casual mask applied.
    '''
    def __call__(self, 
                 inputs_q: jnp.ndarray, 
                 inputs_kv: jnp.ndarray, 
                 mask: Optional[jnp.ndarray] = None, 
                 deterministic: Optional[bool] = None):
        
        key = key if key is not None else inputs_q
        value = value if value is not None else inputs_q

        seq_len = inputs_q.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        mask = mask * causal_mask if mask is not None else causal_mask
        return super().__call__(inputs_q, inputs_kv, mask, deterministic)


class DenseBlock(nn.Module):
    """A 2-layer MLP"""
    init_scale: float = 1.0
    widening_factor: int = 4
    name: Optional[str] = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = jax.nn.initializers.variance_scaling(self.init_scale)

        # fc layer
        x = nn.Dense(self.widening_factor * hiddens,
                     kernel_init=initializer)(x)

        # relu
        x = nn.gelu(x)

        # fc layer
        x = nn.Dense(hiddens,
                     kernel_init=initializer)(x)
        


