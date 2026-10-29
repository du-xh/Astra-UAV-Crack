# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
from torch import nn
import torch.nn.functional as F
from astra.core.alignment import SplitFrequencyFusion
from astra.core.affinity import PagFMAffinity

class PixelAgreementControl(nn.Module):
    """Learn feature agreement; this is not a foreground probability/veto."""

    def __init__(self):
        super().__init__()
        self.affinity = PagFMAffinity(32, 16, after_relu=False, with_channel=False)
        for module in self.affinity.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.strength = nn.Parameter(torch.zeros(()))

    def forward(self, shallow4, semantic8):
        with torch.cuda.amp.autocast(enabled=False):
            agreement = self.affinity(shallow4.float(), semantic8.float())
            return 1.0 + torch.tanh(self.strength) * (2.0 * agreement - 1.0)

class ControlledAlignmentFusion(SplitFrequencyFusion):
    """Unchanged FQ3 arithmetic except optional scaling of its correction."""

    def __init__(self):
        super().__init__('FQ3')

    def forward(self, shallow4, semantic8, multiplier=None):
        if shallow4.shape[1] != 32 or semantic8.shape[1] != 32 or shallow4.shape[-2] != semantic8.shape[-2] * 2 or (shallow4.shape[-1] != semantic8.shape[-1] * 2):
            raise ValueError('Expected projected32 at stride4/8')
        semantic4 = F.interpolate(semantic8, size=shallow4.shape[-2:], mode='bilinear', align_corners=False)
        with torch.cuda.amp.autocast(enabled=False):
            _, enhanced4, aligned4 = self.operator(shallow4.float(), semantic8.float())
            correction = enhanced4 - shallow4.float() + (aligned4 - semantic4.float())
            delta = self.gain * correction
            if multiplier is not None:
                delta = delta * multiplier
        return shallow4 + semantic4 + delta.to(shallow4.dtype)
