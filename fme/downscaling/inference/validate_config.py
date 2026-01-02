import dacite
import argparse
import yaml

from . import InferenceConfig

def validate_config(config_dict: dict) -> InferenceConfig:
    """
    Validate and convert a dictionary to an InferenceConfig dataclass.

    Args:
        config_dict (dict): The configuration dictionary to validate.   
    """

    return dacite.from_dict(
        data_class=InferenceConfig,
        data=config_dict,
        config=dacite.Config(strict=True)
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Validate InferenceConfig from a dictionary.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file (YAML).")
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config_dict = yaml.safe_load(file)

    try:
        config = validate_config(config_dict)
        print("Configuration is valid.")
    except dacite.DaciteError as e:
        print(f"Configuration validation failed: {e}")