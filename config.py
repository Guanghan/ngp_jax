import gin
gin.parse_config_file("configs/default.gin")


@gin.configurable()
class Config:
    """ 
    contain all configurable values;
    leave important or frequently changed ones to gin config files
    """
    # Volume rendering hyper-params
    near_bound = 2.   # near bound of sample space for 3d points
    far_bound = 6.   # far bound of sample space for 3d points
    num_sample_points = 256 # number of points to be sampled across the volume
    epsilon = 1e10  # hyper-params for volume rendering

    # Encoding hyper-params
    apply_positional_encoding = True # original encoding scheme for vanila NeRF
    positional_encoding_dims = 6 # for simpler NeRF
    positional_encoding_dims_xyz = 10  # 3+3*10*2=63 by default, for vanilaNeRF
    positional_encoding_dims_dir = 4  # 3+3*4*2=27 by default, for vanilaNeRF
    positional_encoding_min_degree = 1
    positional_encoding_max_degree = 10
    
    # Network hyper-params
    num_dense_layers = 8  # for MLP
    dense_layer_width = 256 # dimension of dense layers' output space

    # Training hyper-params 
    batch_size = int(1e4)
    lr = 5e-4  # learning rate
    train_epochs = 1000 # number of training epoches
    pot_interval = 100 # epoch interval for plotting results during training