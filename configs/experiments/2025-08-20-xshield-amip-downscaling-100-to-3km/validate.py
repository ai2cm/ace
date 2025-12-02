import dacite
import yaml

from fme.downscaling.inference import InferenceConfig

with open("generate-multivar.yaml") as f:
    generate_config = yaml.safe_load(f)

d = dacite.from_dict(InferenceConfig, data=generate_config)
print("Configuration is valid.")
print(d)
