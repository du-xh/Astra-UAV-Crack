# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
from torch import nn
from mmseg.registry import MODELS
from astra.core.semantic_distillation import UAVH1SemanticSegmentor
from astra.core.repvit import RepVGGDW

@torch.no_grad()
def original_deployment_interface(rep, eps):
    """Fused Wx+b represented by original bias-free conv and canonical eval BN.

    running_var=1-eps,weight=1,mean=0,bias=b, so eval BN is x+b.
    Retaining this existing interface permits the unmodified strict deploy loader.
    """
    if rep.training:
        raise RuntimeError('Fold only eval running-statistics state')
    reference = next(rep.parameters())
    devices = [reference.device.index] if reference.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        folded = rep.fuse()
        conv = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False, device=reference.device, dtype=reference.dtype)
        norm = nn.BatchNorm2d(32, eps=eps).to(device=reference.device, dtype=reference.dtype)
        conv.weight.copy_(folded.weight)
        norm.weight.fill_(1)
        norm.bias.copy_(folded.bias)
        norm.running_mean.zero_()
        norm.running_var.fill_(1 - eps)
        norm.num_batches_tracked.zero_()
    return (conv.requires_grad_(False).eval(), norm.requires_grad_(False).eval())

@MODELS.register_module()
class UAVH1RepDWSegmentor(UAVH1SemanticSegmentor):

    def __init__(self, **kwargs):
        if kwargs.get('transfer_kind') != 'cglu32':
            raise ValueError('H1 CGLU only')
        super().__init__(**kwargs)
        layer = self.detail.depthwise
        conv = layer.conv
        norm = layer.norm
        if conv.weight.shape != (32, 1, 3, 3) or conv.bias is not None or (not isinstance(norm, nn.BatchNorm2d)):
            raise ValueError('Expected original32ch depthwise conv plus BN')
        self.repdw_deployment_eps = norm.eps
        with torch.random.fork_rng(devices=[]):
            rep = RepVGGDW(32)
            with torch.no_grad():
                rep.conv.c.weight.copy_(conv.weight)
                rep.bn.load_state_dict(norm.state_dict(), strict=True)
        layer.conv = rep
        layer.norm = nn.Identity()
        object.__setattr__(self, '_repdw_unfolded', None)
        self.repdw_validation_swaps = 0

    def enter_deployment(self):
        if self._repdw_unfolded is not None:
            raise RuntimeError('Already folded')
        if self.training:
            raise RuntimeError('Expected model.eval before deployment swap')
        layer = self.detail.depthwise
        if not isinstance(layer.conv, RepVGGDW):
            raise RuntimeError('Training DW lost author branches')
        conv, norm = original_deployment_interface(layer.conv, self.repdw_deployment_eps)
        object.__setattr__(self, '_repdw_unfolded', (layer.conv, layer.norm))
        layer.conv = conv
        layer.norm = norm
        self.repdw_validation_swaps += 1

    def exit_deployment(self):
        if self._repdw_unfolded is None:
            raise RuntimeError('No folded deployment to restore')
        self.detail.depthwise.conv, self.detail.depthwise.norm = self._repdw_unfolded
        object.__setattr__(self, '_repdw_unfolded', None)

    def student_state_dict(self, state_dict=None):
        state = super().student_state_dict(state_dict)
        if isinstance(self.detail.depthwise.conv, RepVGGDW):
            conv, norm = original_deployment_interface(self.detail.depthwise.conv, self.repdw_deployment_eps)
            state = {k: v for k, v in state.items() if not k.startswith(('detail.depthwise.conv.', 'detail.depthwise.norm.'))}
            state['detail.depthwise.conv.weight'] = conv.weight.detach()
            state.update({'detail.depthwise.norm.' + k: v for k, v in norm.state_dict().items()})
        return state
