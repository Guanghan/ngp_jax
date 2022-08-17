from pickletools import optimize
from opt import get_opts
import gin

from models import BasicNeRF
from model_utils import generate_rays, perform_volume_rendering

from jax import jit, pmap, value_and_grad, lax, local_device_count, random, device_put_replicated, local_devices
from jax import numpy as jnp

import numpy as np

import optax
from flax.training import train_state, common_utils
from flax.jax_utils import unreplicate

import tqdm
import matplotlib.pyplot as plt


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


def train_and_evaluate(state, train_step_fn, validation_step_fn):
    train_loss_history, train_psnr_history = [], []
    val_loss_history, val_psnr_history = [], []

    for epoch in tqdm(range(config.train_epochs)):
        # shard random number generators
        rng_index, rng_epoch = random.split(random.fold_in(rng, epoch))
        sharded_rngs = common_utils.shard_prng_key(rng_epoch)

        # create training batch 
        train_index = random.randint(rng_index, (n_devices,), minval=0, maxval=len(train_rays))
        train_batch = train_rays[tuple(train_index), ...],\
                      train_images[tuple(train_index), ...]
        
        # perform training step
        train_loss, train_psnr, state = train_step_fn(state, train_batch, sharded_rngs)
        avg_train_loss = np.asarray(np.mean(train_loss))
        avg_train_psnr = np.asarray(np.mean(train_psnr))
        train_loss_history.append(avg_train_loss)
        train_psnr_history.append(avg_train_psnr)
        
        # logging
        print("Train loss @ epoch {}: {}".format(epoch, avg_train_loss))
        print("Train psnr @ epoch {}: {}".format(epoch, avg_train_psnr))

        # perform validation step
        validation_state = unreplicate(state)
        rgb, depth, val_psnr, val_loss = validation_step_fn(validation_state)
        avg_val_loss = np.asarray(np.mean(val_loss))
        avg_val_psnr = np.asarray(np.mean(val_psnr))
        val_loss_history.append(avg_val_loss)
        val_psnr_history.append(avg_val_psnr)
        
        # logging
        print("Val loss @ epoch {}: {}".format(epoch, avg_train_loss))
        print("Val psnr @ epoch {}: {}".format(epoch, avg_train_psnr))

        # plot result at every plot interval
        if epoch % config.plot_interval == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
            ax1.imshow(rgb)
            ax1.set_title("Predicted RGB at epoch: {}".format(epoch))
            ax1.axis('off')
            ax2.imshow(depth)
            ax2.set_title('Predicted RGB at epoch: {}'.format(epoch))
            ax2.axis('off')
            plt.show()
    
    inference_state = unreplicate(state)
    history = {
        'train_loss': train_loss_history,
        'train_psnr': train_psnr_history,
        'val_loss': val_loss_history,
        'val_psnr': val_psnr_history
    }
    return state, inference_state, history





if __name__ == '__main__':
    args = get_opts()
    gin.parse_config_file(args.gin_config_path)

    from config import Config
    config = Config()

    # Apply the transform jax.pmap on the train_step to parallelize it on XLA devices
    # While vmap vectorizes a function by adding a batch dimension to every primitive operation 
    # in the function, pmap replicates the function and executes each replica 
    # on its own XLA device in parallel.
    parallelized_train_step = pmap(train_step, axis_name="batch")

    data = np.load("tiny_nerf_data.npz")
    images = data["images"]
    poses = data["poses"]
    focal = float(data["focal"])

    _, img_ht, img_wid, _ = images.shape
    
    train_images, train_poses = images[:100], poses[:100]
    val_image, val_pose = images[101], poses[101]

    # create train and validation rays
    train_rays = np.stack(list(map(lambda x: generate_rays(img_ht, img_wid, focal, x), train_poses)))
    val_rays = generate_rays(img_ht, img_wid, focal, val_pose)

    # number of devices
    n_devices = local_device_count()

    # rand number generation
    key, rng = random.split(random.PRNGKey(0))

    # init the model
    model, params = init_model(key, (img_ht*img_wid, 3))

    # optimizer
    optimizer = optax.adam(learning_rate=config.lr)

    # training state
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # transfer arrays in the state to specified devices and form ShardedDeviceArrays
    state = device_put_replicated(state, local_devices())


