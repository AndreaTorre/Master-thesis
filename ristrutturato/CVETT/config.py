# Wrapper verso il config unico.
# Serve per mantenere compatibili i vecchi import: import config

import importlib.util
from pathlib import Path

_common_config_path = Path("/home/atorre/UTSP/unione/git/UTSP/common/config.py")

_spec = importlib.util.spec_from_file_location("_utsp_common_config", _common_config_path)
_common_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common_config)

for _name in dir(_common_config):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_common_config, _name)
