import numpy as np
import jax.numpy as jnp
from jax.random import uniform
from jax import lax, vmap
import flax.linen as nn

from einops import rearrange, reduce, repeat

from config import Config
config = Config()


def generate_rays(ht, wid, focal, pose):
    """
    Given a pose, generate a grid of rays corresponding\
        to the dimensions of the image set to render
    Reference:
        https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf
        https://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf
        https://www.cse.psu.edu/~rtc12/CSE486/lecture13.pdf
        https://jsantell.com/3d-projection/
    """
    # (1) img_coords -> film_coords -> cam_coords
    # Create a 2D rectangular grid for the rays corresponding to image dims
    i, j = jnp.meshgrid(np.arange(wid), np.arange(ht), indexing="xy")
    offset_x = offset_y = 0.5
    transformed_i = (i - wid * offset_x) / focal 
    transformed_j = -(j - ht * offset_y) / focal 
    # Create the unit vectors corresponding to ray directions
    k = -jnp.ones_like(i) # z-axis coordinates
    directions = jnp.stack([transformed_i, transformed_j, k], axis=-1)

    # (2) cam_coords -> canonical_world_coords
    # Get rotation and translation matrices from the extrinsic params (transposed 4x4 homo trans matrix)
    rotation_matrix, translation_matrix = pose[:3, :3], pose[:3, -1]
    # Matrix multiplication for each pixel (i, j).  rotation_matrix * 3d_coords
    ray_directions = jnp.einsum("i j l, k l -> i j k", directions, rotation_matrix)
    ray_origins = repeat(translation_matrix, 'k -> i j k', i=wid, j=ht, k=3)
    
    return jnp.stack([ray_origins, ray_directions])


def compute_3d_points(ray_origins, ray_directions, rand_num_generator=None):
    """
    Compute 3d query points for volumetric rendering
    """
    # sample space to parametrically compute the ray points
    t_vals = jnp.linspace(config.near_bound, config.far_bound, config.num_sample_points)

    # inject a random noise into the sample space to make it continuous
    if rand_num_generator is not None:
        t_shape = ray_origins.shape[:-1] + (config.num_sample_points,)
        noise = uniform(rand_num_generator, t_shape) * (config.far_bound - config.near_bound) / config.num_sample_points
        t_vals += noise
    
    # compute the ray traversal points using: r(t) = o + d*t
    ray_origins = rearrange(ray_origins, "i j k -> i j 1 k")
    ray_directions = rearrange(ray_directions, "i j k -> i j 1 k")
    t_vals_flat = rearrange(t_vals, "n -> n 1")
    points = ray_origins + ray_directions * t_vals_flat

    return points, t_vals


def compute_radiance_field(model, points):
    """
    Compute radiance field
    """
    # compared to jax.vmap, lax.map will apply the function element by element\
    # for reduced memory usage
    # points: (w, h, samples, 3) -> (-1, batch, 3)
    model_output = lax.map(model, jnp.reshape(points, (-1, config.batch_size, 3))) 
    
    # model_output:(-1, batch, 4) -> radiance_field: (w, h, samples, 4)
    radiance_field = jnp.reshape(model_output, points.shape[:-1] + (4,))
    
    opacities = nn.relu(radiance_field[..., 3])
    colors = nn.sigmoid(radiance_field[..., :3])

    return opacities, colors







