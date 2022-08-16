import jax, flax, optax
from jax import lax
import flax.linen as nn

import numpy as np
import jax.numpy as jnp

from typing import Any, Callable

from config import Config
config = Config()

class BasicNeRF(nn.Module):
    dtype: Any = jnp.float32
    precision: Any = lax.Precision.DEFAULT
    apply_positional_concoding: bool = config.apply_positional_encoding