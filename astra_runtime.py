"""Portable loading of the original student, without constructing its teacher."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor'))


def setup():
    import cv2
    import torch
    from mmseg.utils import register_all_modules
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    register_all_modules(init_default_scope=True)


def config(name='astra_deploy.py'):
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    value = Config.fromfile(str(ROOT / 'configs' / name), lazy_import=False)
    if value.get('custom_imports'):
        import_modules_from_strings(**value.custom_imports)
    return value


def load_student(checkpoint, device='cuda:0'):
    import torch
    from mmengine.model import revert_sync_batchnorm
    from mmseg.registry import MODELS
    setup()
    model = MODELS.build(config().model)
    stored = torch.load(str(checkpoint), map_location='cpu')
    model.load_state_dict(stored['state_dict'], strict=True)
    count = sum(p.numel() for p in model.parameters())
    if count != 15310826:
        raise RuntimeError('Student deployment graph mismatch: %d parameters' % count)
    return revert_sync_batchnorm(model).to(device).eval().requires_grad_(False)
