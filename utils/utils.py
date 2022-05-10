import yaml
import os 
from from_root import from_root

def read_config(config_path=os.path.join(from_root(),"config.yaml")):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)

    return content


if __name__ == "__main__":
    print(read_config())