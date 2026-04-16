# --------------------------------------------------------------------------------
# Config Loader
# Loading system policy weights
# --------------------------------------------------------------------------------

import json
from pathlib import Path

CONFIG_PATH = Path("app/config/system_config.json")

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)