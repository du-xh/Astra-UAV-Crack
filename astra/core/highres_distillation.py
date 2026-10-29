# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
from mmseg.registry import MODELS
from astra.core.teacher_features import UAVCGLULocalDistillSegmentor
from astra.core.highres_loss import UAVHighResolutionReconstruction

@MODELS.register_module()
class UAVCGLUHighResDistillSegmentor(UAVCGLULocalDistillSegmentor):

    def __init__(self, feature_target_mode, **kwargs):
        if kwargs.get('contrast_enabled') or kwargs.get('feature_contrast_weight') != 0:
            raise ValueError('Stage30 isolates reconstruction, with contrast disabled')
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            self.feature_distill = UAVHighResolutionReconstruction(feature_target_mode)
