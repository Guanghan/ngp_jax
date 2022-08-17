import dataclasses
import gin


@gin.configurable()
@dataclasses.dataclass
class Config:
    """ 
    contain all configurable values;
    leave important or frequently changed ones to gin config files
    """
    # Volume rendering hyper-params
    near_bound: float = 2.   # near bound of sample space for 3d points
    far_bound: float = 6.   # far bound of sample space for 3d points
    num_sample_points: int = 256 # number of points to be sampled across the volume
    epsilon: float = 1e10  # hyper-params for volume rendering

    # Encoding hyper-params
    apply_positional_encoding: bool = True # original encoding scheme for vanila NeRF
    positional_encoding_dims: int = 6 # for simpler NeRF
    positional_encoding_dims_xyz: int = 10  # 3+3*10*2=63 by default, for vanilaNeRF
    positional_encoding_dims_dir: int = 4  # 3+3*4*2=27 by default, for vanilaNeRF
    positional_encoding_min_degree: int = 1
    positional_encoding_max_degree: int = 10
    
    # Network hyper-params
    num_dense_layers: int = 8  # for MLP
    dense_layer_width: int = 256 # dimension of dense layers' output space

    # Training hyper-params 
    batch_size: int = int(1e4)
    lr: float = 5e-4  # learning rate
    train_epochs: int = 1000 # number of training epoches
    pot_interval: int = 100 # epoch interval for plotting results during training

    # visualization
    plot_interval = 5  # plot validation result after this number of epoches