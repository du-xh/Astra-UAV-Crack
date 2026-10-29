"""Load the separately obtained author core; no upstream license is inferred.

The student inference path does not import this training-only module.
See THIRD_PARTY_NOTICES.md and tools/fetch_eakd.py before training.
"""
import importlib.util
import os
from pathlib import Path
REPOSITORY = Path(__file__).resolve().parents[2]

path = Path(os.environ.get('ASTRA_EAKD_CORE', str(REPOSITORY/'external/EA_KD_author.py')))
if not path.is_file():
    raise RuntimeError('EA-KD training core is external. Obtain it with '
                       'python tools/fetch_eakd.py after reviewing its upstream terms.')
spec = importlib.util.spec_from_file_location('astra_external_eakd_author', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
get_w = module.get_w
kd_loss = module.kd_loss
