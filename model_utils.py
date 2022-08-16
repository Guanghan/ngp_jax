from gettext import translation
import numpy as np
import jax.numpy as jnp
from einops import rearrange, reduce, repeat


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
    transformed_i = (i - wid * offset_x) / focal # Normalize the x-axis coordinates
    transformed_j = -(j - ht * offset_y) / focal # Normalize the y-axis coordinates
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