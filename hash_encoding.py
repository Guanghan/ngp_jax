from flax import linen as nn
from jax import numpy as jnp
import jax
from typing import Any

from config import Config
config = Config()


class HashEmbedder(nn.Module):
    bbox3D: Any = config.bbox3D    # only deal with points within bbox3D
    
    n_levels: int = 16             # L
    n_feats_per_level: int = 2     # F
    finest_res: int = 512          # N_max
    coarse_res: int = 16           # N_min
    log2_hash_sz: int = 19         # log2(T) -> T = 2^19

    output_dim = n_levels * n_feats_per_level # L*F
    b = jnp.exp((jnp.log(finest_res) - jnp.log(coarse_res)) / (n_levels-1)) # equation (3)
    embeddings = [nn.Embed(num_embeddings= 2**log2_hash_sz, 
                           features= n_feats_per_level, 
                           param_dtype= jnp.float16)]

    # initialize embeddings (how?) 
    for i in range(n_levels):
       jax.nn.initializers.glorot_uniform(embeddings[i]) 

    @nn.compact
    def __call__(self, input_points):
        # input_points: Bx3
        embeds_all = []
        for i in range(self.n_levels):
            # get the resolution for this level
            resolution = jnp.floor(self.coarse_res * self.b**i)
            # get the 8 voxel vertices at this level
            vox_min_vert, vox_max_vert, hashed_vox_indices = self.get_voxel_vertices(input_points,
                                                                                    self.bbox3D, 
                                                                                    resolution, 
                                                                                    self.log2_hash_sz)
            # get 8 vertex feature embeddings with indices from the hash encoding
            vox_embeds = self.embeddings[i](hashed_vox_indices)
            # interpolate to get the point features from 8 vertices
            pt_embed = self.trilinear_interp(input_points, vox_min_vert, vox_max_vert, vox_embeds)
            # combine features from all levels
            embeds_all.append(pt_embed)

        return jnp.concatenate(embeds_all, axis=-1)


    @jax.jit
    def trilinear_interp(self, x, voxel_min_vert, voxel_max_vert, voxel_embeds):
        '''
        x:  Bx3
        voxel_min_vert:  Bx3
        voxel_max_vert:  Bx3
        voxel_embeds:    Bx8x2
        '''
        pass


    def get_voxel_vertices(self, xyz, bbox3D, resolution, log2_hash_sz):
        '''
        xyz: 3D coordinates of sample points, shape Bx3
        bbox3D: min and max x,y,z coordinates of a 3D object bbox
        resolution: number of voxels per axis
        '''
        pass