from opt import get_opts
import gin

if __name__ == '__main__':
    args = get_opts()
    gin.parse_config_file(args.gin_config_path)
