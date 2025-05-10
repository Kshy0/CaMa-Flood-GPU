from omegaconf import OmegaConf
from CMF_GPU.utils.Parameter import DefaultGlobalCatchment
def generate_parameters(config_file):
    config = OmegaConf.load(config_file)
    param_set = DefaultGlobalCatchment(config)
    param_set.build_model_input_pipeline()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CMF_GPU simulation.")
    parser.add_argument("config_file", type=str, help="Path to the config.yaml file")
    args = parser.parse_args()
    generate_parameters(args.config_file)
