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
    i, j = np.meshgrid(np.arange(wid), np.arange(ht), indexing="xy")
    offset_x = offset_y = 0.5
    transformed_i = (i - wid * offset_x) / focal 
    transformed_j = -(j - ht * offset_y) / focal 
    # Create the unit vectors corresponding to ray directions
    k = -np.ones_like(i) # z-axis coordinates
    directions = np.stack([transformed_i, transformed_j, k], axis=-1)

    # (2) cam_coords -> canonical_world_coords
    # Get rotation and translation matrices from the extrinsic params (transposed 4x4 homo trans matrix)
    rotation_matrix, translation_matrix = pose[:3, :3], pose[:3, -1]
    # Matrix multiplication for each pixel (i, j).  rotation_matrix * 3d_coords
    ray_directions = np.einsum("i j l, k l -> i j k", directions, rotation_matrix)
    ray_origins = repeat(translation_matrix, 'k -> i j k', i=wid, j=ht, k=3)
    
    return np.stack([ray_origins, ray_directions])


def compute_3d_points(ray_origins, ray_directions, rand_num_generator=None):
    """
    Compute the points along the ray by adding the ray origin to the ray direction multiplied by the
    sample space
    
    :param ray_origins: the origin of the ray
    :param ray_directions: (i, j, 3)
    :param rand_num_generator: a random number generator that is used to inject noise into the sample
    space
    :return: The points along the ray and the t_vals
    """
    # sample space to parametrically compute the ray points
    t_vals = np.linspace(config.near_bound, config.far_bound, config.num_sample_points)

    # inject a random noise into the sample space to make it continuous
    '''
    # needs to debug
    if rand_num_generator is not None:
        t_shape = ray_origins.shape[:-1] + (config.num_sample_points,)
        noise = uniform(rand_num_generator, t_shape) * (config.far_bound - config.near_bound) / config.num_sample_points
        t_vals = t_vals + noise
    '''
    
    # compute the ray traversal points using: r(t) = o + d*t
    ray_origins = rearrange(ray_origins, "i j k -> i j 1 k")
    ray_directions = rearrange(ray_directions, "i j k -> i j 1 k")
    t_vals_flat = rearrange(t_vals, "n -> n 1")
    points = ray_origins + ray_directions * t_vals_flat

    return points, t_vals


def compute_radiance_field(model, points):
    """
    Compute radiance field:
    Take the points, reshape them to be a list of batches, apply the model to each batch, and then
    reshape the output back to the original shape
    
    :param model: the model we trained
    :param points: (w, h, samples, 3)
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


def compute_adjacent_distances(t_vals, ray_directions):
    """
    It computes the distance between adjacent points along a ray
    
    :param t_vals: The t values of the intersections of the rays with the mesh
    :param ray_directions: The direction of each ray
    :return: The distances between the points on the ray that intersect with the object.
    """
    distances = t_vals[..., 1:] - t_vals[..., :-1]
    distances = jnp.concatenate([
        distances,
        np.broadcast_to([config.epsilon], distances[..., :1].shape)
    ], axis=-1)

    # Multiply each distance by the norm of its corresponding direction ray\
    # to convert to real world distance (accounts for non-unit directions)
    distances = distances * jnp.linalg.norm(ray_directions[..., None, :], axis=-1)
    return distances


def compute_rgb_weights(opacities, distances):
    """
    :param opacities: The opacity of each voxel
    :param distances: the distance from the camera to each point in the volume
    :return: The color alpha composition weights for the samples.
    """
    density = jnp.exp(-opacities * distances)
    alpha = 1.0 - density
    clipped_diff = jnp.clip(1.0 - alpha, 1e-10, 1.0)
    transmittance = jnp.cumprod(
        jnp.concatenate(
            [jnp.ones_like(clipped_diff[..., :1]), clipped_diff[..., :-1]], axis=-1
        ), 
        axis = -1
    )
    return alpha * transmittance


def perform_volume_rendering(model, ray_origins, ray_directions, rand_num_generator=None):
    """
    Volume rendering
    """
    # compute 3d query points
    #print("compute 3d points")
    points, t_vals = compute_3d_points(ray_origins, ray_directions, rand_num_generator)

    # get distances between adjacent intervals along sample space
    #print("compute adjacent distances")
    distances = compute_adjacent_distances(t_vals, ray_directions)
    
    # get color and opacities from MLPs
    #print("compute radiance field")
    opacities, colors = compute_radiance_field(model, points)

    # compute weight for the RGB color of each sample along each ray
    #print("compute rgb weights")
    rgb_weights = compute_rgb_weights(opacities, distances)

    # compute weighted RGB color of each sample along each ray
    #print("compute rgb map")
    rgb_map = jnp.sum(rgb_weights[..., None] * colors, axis=-2)

    # compute the estimated depth map
    #print("compute depth map")
    depth_map = jnp.sum(rgb_weights * t_vals, axis = -1)

    # sum of weights along each ray; value in range [0, 1]
    #print("compute acc map")
    acc_map = jnp.sum(rgb_weights, axis=-1)

    # disparity map: inverse of the depth map
    #print("compute disparity map")
    disparity_map = 1. / jnp.maximum(1e-10, depth_map / acc_map)

    return rgb_map, depth_map, acc_map, disparity_map, opacities


