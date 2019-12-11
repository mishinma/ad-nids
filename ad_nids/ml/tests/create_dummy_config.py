
import logging
from pathlib import Path

from ad_nids.config import create_configs


loglevel = getattr(logging, "DEBUG", None)
logging.basicConfig(level=loglevel)

data_path = Path('data/processed')
params_root_path = Path('data/params')
config_out_path = Path('data/config')

for params_path in params_root_path.iterdir():
    create_configs(data_path, config_out_path, params_path)
