from opt import get_opts
import gin

from models import BasicNeRF
from model_utils import perform_volume_rendering

from jax import jit, pmap, value_and_grad, lax
from jax import numpy as jnp


def init_model(key, input_pts_shape):
    # create model
    model = BasicNeRF()

    # init model
    init_params = jit(model.init)({"params":key}, jnp.ones(input_pts_shape))

    return model, init_params["params"]


def train_step(state, batch, rng):
    """
    Training Step:
    state is comb of model state and optimizer state
    """
    inputs, targets = batch

    # compute loss in a stateless manner to save memory
    def loss_fn(params):
        ray_origins, ray_directions = inputs
        model_fn = lambda x: state.apply_fn({"params": params}, x)
        pred_rgbs, *_ = perform_volume_rendering(model_fn, ray_origins, ray_directions, rng)
        return jnp.mean( (pred_rgbs - targets)**2 )

    # get loss value and gradients    
    train_loss, gradients = value_and_grad(loss_fn)(state.params)

    # get averaged train loss
    train_loss = jnp.mean(train_loss)

    # compute all-reduce mean on gradients over the pmapped axis
    gradients = lax.pmean(gradients, axis_name="batch")

    # update model params & optimizer state
    new_state = state.apply_gradients(grads=gradients)

    # compute PSNR
    train_psnr = -10.0 * jnp.log(train_loss) / jnp.log(10.0)


@jit
def validation_step(state, val_rays, val_img):
    """
    Validation
    """
    model_fn = lambda x: state.apply_fn({"params": state.params}, x)
    ray_origins, ray_directions = val_rays

    pred_rgb, pred_depth, *_ = perform_volume_rendering(model_fn, ray_origins, ray_directions)

    val_loss = jnp.mean( (pred_rgb - val_img)**2 )
    val_psnr = -10.0 * jnp.log(val_loss) / jnp.log(10.0)
    return pred_rgb, pred_depth, val_psnr, val_loss


if __name__ == '__main__':
    args = get_opts()
    gin.parse_config_file(args.gin_config_path)

    # Apply the transform jax.pmap on the train_step to parallelize it on XLA devices
    # While vmap vectorizes a function by adding a batch dimension to every primitive operation 
    # in the function, pmap replicates the function and executes each replica 
    # on its own XLA device in parallel.
    parallelized_train_step = pmap(train_step, axis_name="batch")

