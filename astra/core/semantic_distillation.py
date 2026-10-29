# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from astra.core.highres_distillation import UAVCGLUHighResDistillSegmentor

def reconstruction_semantic_kl(reconstructed, target, teacher_head, valid, temperature=2.0):
    """Frozen head keeps input autograd; only the teacher target is detached."""
    if teacher_head.training or any((p.requires_grad for p in teacher_head.parameters())):
        raise RuntimeError('Semantic head must be frozen/eval')
    if reconstructed.shape != target.shape or reconstructed.shape[1] != 512:
        raise ValueError('Semantic reconstruction512 target contract changed')
    with torch.cuda.amp.autocast(enabled=False):
        prediction = teacher_head.cls_seg(reconstructed.float())
        with torch.no_grad():
            reference = teacher_head.cls_seg(target.detach().float())
        prediction = F.interpolate(prediction, size=valid.shape[-2:], mode='bilinear', align_corners=False)
        reference = F.interpolate(reference, size=valid.shape[-2:], mode='bilinear', align_corners=False)
        pointwise = F.kl_div(F.log_softmax(prediction / temperature, dim=1), F.softmax(reference / temperature, dim=1), reduction='none').sum(1) * temperature ** 2
        if not valid.any():
            raise ValueError('Empty valid semantic region')
        return pointwise[valid].mean()

@MODELS.register_module()
class UAVH1SemanticSegmentor(UAVCGLUHighResDistillSegmentor):

    def __init__(self, semantic_weight=0.0, semantic_temperature=2.0, **kwargs):
        if kwargs.get('feature_target_mode') != 'pooled_target':
            raise ValueError('Stage35 holds the H1 pooled teacher target fixed')
        super().__init__(**kwargs)
        self.semantic_weight = float(semantic_weight)
        self.semantic_temperature = float(semantic_temperature)
