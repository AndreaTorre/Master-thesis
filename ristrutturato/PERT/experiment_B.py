# Wrapper verso common/experiment_B.py

import sys
import importlib.util
from pathlib import Path

_common_dir = Path("/home/atorre/UTSP/unione/git/UTSP/common")
_common_file = _common_dir / "experiment_B.py"

if str(_common_dir) not in sys.path:
    sys.path.insert(0, str(_common_dir))

_spec = importlib.util.spec_from_file_location("_utsp_common_experiment_B", _common_file)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name in dir(_mod):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_mod, _name)
