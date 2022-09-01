from flax import linen as nn
from jax import numpy as jnp
import jax, flax
from typing import Any
import pdb
from jax import jit

from config import Config
config = Config()


class HashEmbedder(nn.Module):
    n_levels: int = 16             # L
    n_feats_per_level: int = 2     # F
    finest_res: int = 512          # N_max
    coarse_res: int = 16           # N_min
    log2_hash_sz: int = 19         # log2(T) -> T = 2^19
    BOX_OFFSETS: jnp.array = jnp.array([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]])

    @nn.compact
    def __call__(self, input_points):
        output_dim = self.n_levels * self.n_feats_per_level # L*F
        b = jnp.exp((jnp.log(self.finest_res) - jnp.log(self.coarse_res)) / (self.n_levels-1)) # equation (3)

        # Construct embeddings 
        hash_embeddings = []
        for i in range(self.n_levels):
            # Embed: A parameterized function from integers [0, n) to d-dimensional vectors.
            cur_embed = nn.Embed(num_embeddings= 2**self.log2_hash_sz, 
                            features= self.n_feats_per_level, 
                            embedding_init = jax.nn.initializers.uniform(scale=1e-4),
                            param_dtype= jnp.float32)  # float16 in paper
            jax.nn.initializers.glorot_uniform(cur_embed) 
            hash_embeddings.append(cur_embed)

        # input_points: Bx3
        bbox3D = config.bbox3D    # only deal with points within bbox3D
        embeds_all = []
        for i in range(self.n_levels):
            # get the resolution for this level
            resolution = jnp.floor(self.coarse_res * b**i)
            # get the 8 voxel vertices at this level
            vox_min_vert, vox_max_vert, hashed_vox_indices = self.get_voxel_vertices(input_points,
                                                                                    bbox3D, 
                                                                                    resolution, 
                                                                                    self.log2_hash_sz)
            
            # get 8 vertex feature embeddings with indices from the hash encoding
            vox_embeds = hash_embeddings[i](hashed_vox_indices)
            # interpolate to get the point features from 8 vertices
            pt_embed = self.trilinear_interp(input_points, vox_min_vert, vox_max_vert, vox_embeds)
            # combine features from all levels
            embeds_all.append(pt_embed)

        return jnp.concatenate(embeds_all, axis=-1)

    @jit
    def trilinear_interp(self, x, vox_min_vert, vox_max_vert, vox_embeds):
        '''
        x:  Bx3
        voxel_min_vert:  Bx3
        voxel_max_vert:  Bx3
        voxel_embeds:    Bx8x2
        '''
        # Reference: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - vox_min_vert) / (vox_max_vert - vox_min_vert) # Bx3

        # step 1
        # index representation: 
        # 0->000, 1->001, 2->010, 3->011, 4->100. 5->101 6->110. 7->111
        c00 = vox_embeds[:,0]*(1-weights[:,0][:,None]) + vox_embeds[:,4]*weights[:,0][:,None]
        c01 = vox_embeds[:,1]*(1-weights[:,0][:,None]) + vox_embeds[:,5]*weights[:,0][:,None]
        c10 = vox_embeds[:,2]*(1-weights[:,0][:,None]) + vox_embeds[:,6]*weights[:,0][:,None]
        c11 = vox_embeds[:,3]*(1-weights[:,0][:,None]) + vox_embeds[:,7]*weights[:,0][:,None]
        
        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    @jit
    def get_voxel_vertices(self, xyz, bbox3D, resolution, log2_hash_sz):
        '''
        xyz: 3D coordinates of sample points, shape Bx3
        bbox3D: min and max x,y,z coordinates of a 3D object bbox
        resolution: number of voxels per axis
        '''
        box_min, box_max = bbox3D

        # if not jnp.all(xyz < box_max) or not jnp.all(xyz > box_min):
        #     pdb.set_trace()
        #     xyz = jnp.clamp(xyz, min=box_min, max=box_max)
        
        grid_size = (box_max - box_min) / resolution

        bottom_left_idx = jnp.floor((xyz - box_min)/grid_size).astype(int)

        vox_min_vert = bottom_left_idx * grid_size + box_min
        vox_max_vert = vox_min_vert + jnp.array([1.0, 1.0, 1.0]) * grid_size

        vox_indices = jnp.expand_dims(bottom_left_idx, 1) + self.BOX_OFFSETS
        hashed_vox_indices = self.hash(vox_indices, log2_hash_sz)

        return vox_min_vert, vox_max_vert, hashed_vox_indices
    
    @jit
    def hash(self, coords, log2_hash_sz):
        '''
        coords: 3D coordinates in shape Bx3
        log2T: logarithm of T w.r.t 2
        Reference: http://www.beosil.com/download/CollisionDetectionHashing_VMV03.pdf
        '''
        x, y, z = coords[...,0], coords[...,1], coords[...,2]
        return ((1<<log2_hash_sz)-1) & (x*73856093 ^ y*19349663 ^ z*83492791)