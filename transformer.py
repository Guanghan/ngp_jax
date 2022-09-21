import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
#import SelfAttention, MultiHeadDotProductAttention, Module

from typing import Optional, Mapping

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
        return x


def layer_norm(x: jnp.ndarray,
               name: Optional[str] = None)-> jnp.ndarray:
    return nn.LayerNorm(use_bias=True, use_scale=True)(x)


class Transformer(nn.Module):
    """
    A transformer stack. Each block includes:
    - a norm layer
    - a self-attention layer
    - two dropout layers
    - two normalization layers
    - two skip connections
    - a 2-layered Dense block
    """
    num_heads: int = 1
    num_layers: int = 1
    dropout_rate: float = 0.5
    name: Optional[str] = None

    def __call__(self, h, mask, is_training):
        '''
        Args: 
            h: Inputs, jnp.ndarray, [B, T, H]
            mask: Padding mask, jnp.ndarray, [B, T]
            is_training: bool
        '''
        init_scale = 2. / self.num_layers
        dropout_rate = self.dropout_rate if is_training else 0.
        
        initializer = jax.nn.initializers.variance_scaling(init_scale)

        if mask is not None:
            mask = mask[:, None, None, :]
        
        for i in range(self.num_layers):
            h_norm = layer_norm(h, name=f'h{i}_ln_1')
            h_attn = nn.SelfAttention(num_heads=self.num_heads,
                                      qkv_features=64,
                                      kernel_init=initializer,
                                      bias_init=initializer,
                                      name = f'h{i}_attn')(h_norm, mask=mask)
            h_attn = nn.Dropout(rate=dropout_rate)(h)

            h = h + h_attn # skip connection

            h_norm = layer_norm(h, name=f'h{i}_ln_2')
            h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
            h_dense = nn.Dropout(rate=dropout_rate)(h_dense)

            h = h + h_dense # skip connection
        h = layer_norm(h, name='ln_f')
        return h



class FlaxEmbeddings(nn.Module):
    def setup(self):
        embed_init =  jax.nn.initializers.glorot_normal()
        self.token_embedding_map = nn.Embed(num_embeddings=vocab_size, d_model)
        
        # positional embeddings are trainable (as opposed to positional encodings in BERT that are fixed)
        self.positional_embeds = 
                                    )

    def embeddings(self, data: Mapping[str, jnp.ndarray], vocab_size: int):
        """
        Ref: 
        https://theaisummer.com/positional-embeddings/
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_flax_vit.py
        """
        tokens = data['obs']
        input_mask = jnp.greater(tokens, 0)
        seq_len = tokens.shape[1]

        # embed the input tokens and positions
        token_embeds = self.token_embedding_map(tokens)
        input_embeds = token_embeds + self.positional_embeds

        return input_embeds, input_mask


                                   
        


