from flax import linen as nn


class HashEmbedder(nn.Module):
    n_levels: int = 16             # L
    n_feats_per_level: int = 2     # F
    finest_res: int = 512          # N_max
    coarse_res: int = 16           # N_min
    log2_hash_sz: int = 19         # log2(T) -> T = 2^19


    @nn.compact
    def __call__(self, input_points):
        return input_points