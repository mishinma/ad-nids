from datetime import datetime
from pathlib import Path


def get_log_dir(log_root_dir, config_name):
    uniq_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f') + '_' + config_name
    log_dir = Path(log_root_dir) / uniq_name
    log_dir = log_dir.resolve()
    return log_dir
