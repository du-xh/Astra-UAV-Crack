# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
from torch import nn
import torch.nn.functional as F
from mmengine.logging import MMLogger
from mmseg.registry import MODELS
from astra.core.cached_distillation import UAVCachedKD20261015Segmentor
from astra.vendor.efficientvit.models.nn.ops import ConvLayer
from astra.vendor.efficientvit.models.utils import list_sum

class DetailResidual(nn.Module):
    """Project native stride4/8 features before inexpensive residual decoding."""

    def __init__(self, dynamic=False):
        super().__init__()
        self.low_project = ConvLayer(96, 32, 1, norm='bn2d', act_func='hswish')
        self.shallow_project = ConvLayer(48, 32, 1, norm='bn2d', act_func='hswish')
        self.depthwise = ConvLayer(32, 32, 3, groups=32, norm='bn2d', act_func='hswish')
        self.pointwise = ConvLayer(32, 32, 1, norm='bn2d', act_func='hswish')
        self.residual_logits = nn.Conv2d(32, 2, 1, bias=True)
        nn.init.zeros_(self.residual_logits.weight)
        nn.init.zeros_(self.residual_logits.bias)
        self.upsampler = None

    def forward(self, head8, shallow4):
        low = self.low_project(head8)
        if self.upsampler is None:
            low = F.interpolate(low, size=shallow4.shape[-2:], mode='bilinear', align_corners=False)
        else:
            low = self.upsampler(low)
        detail = low + self.shallow_project(shallow4)
        return self.residual_logits(self.pointwise(self.depthwise(detail)))

@MODELS.register_module()
class UAVInnovation20261016Segmentor(UAVCachedKD20261015Segmentor):

    def __init__(self, variant='C0', initial_checkpoint=None, **kwargs):
        if variant != 'D4-B':
            raise ValueError('Unknown Stage12 arm: ' + variant)
        super().__init__(**kwargs)
        self.variant = variant
        self.initial_checkpoint = initial_checkpoint
        self.continuation_load_report = None
        if initial_checkpoint is not None:
            obj = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)
            self.load_state_dict(obj['state_dict'], strict=True)
            self.continuation_load_report = dict(path=initial_checkpoint, strict=True, source_iteration=obj.get('meta', {}).get('iter'), tensors=len(obj['state_dict']))
            del obj
        self.detail = None
        with torch.random.fork_rng(devices=[]):
            if variant in ('D4-B', 'D4-D'):
                self.detail = DetailResidual(dynamic=variant == 'D4-D')

    def init_weights(self):
        if self._is_init:
            return
        super().init_weights()
        MMLogger.get_current_instance().info('Stage12 arm=%s; strict continuation=%s', self.variant, self.continuation_load_report)

    def head_with_middle(self, features):
        """Author DAGBlock operations with an explicit return of its middle feature."""
        head = self.network.head
        if head.merge != 'add' or head.output_keys != ['segout']:
            raise ValueError('Only the frozen author B2 ADE semantic head is supported')
        values = [op(features[key]) for key, op in zip(head.input_keys, head.input_ops)]
        middle = list_sum(values)
        if head.post_input is not None:
            middle = head.post_input(middle)
        middle = head.middle(middle)
        return (head.output_ops[0](middle), middle)
