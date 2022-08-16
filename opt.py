import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gin_config_path', type=str,
                        default='configs/default.gin',
                        help='choose which gin config file to use')
    return parser.parse_args()