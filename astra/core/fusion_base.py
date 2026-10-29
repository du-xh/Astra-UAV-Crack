# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
from torch import nn
import torch.nn.functional as F
from astra.core.freqfusion import FreqFusion

class ConservativeFrequencyFusion(nn.Module):
    """Keep DC1's sum and learn the strength of the author feature correction."""

    def __init__(self):
        super().__init__()
        self.operator = FreqFusion(hr_channels=32, lr_channels=32, compressed_channels=16, scale_factor=1, lowpass_kernel=5, highpass_kernel=3, up_group=1, encoder_kernel=3, upsample_mode='nearest', align_corners=False, feature_resample=True, feature_resample_group=4, comp_feat_upsample=True, use_high_pass=True, use_low_pass=True, hr_residual=True, semi_conv=True, hamming_window=False, feature_resample_norm=True)
        self.gain = nn.Parameter(torch.full((1, 32, 1, 1), 0.01))

    def forward(self, shallow4, semantic8):
        if shallow4.shape[1] != 32 or semantic8.shape[1] != 32 or shallow4.shape[-2] != semantic8.shape[-2] * 2 or (shallow4.shape[-1] != semantic8.shape[-1] * 2):
            raise ValueError('Frequency fusion expects projected32 channels at stride4/8')
        semantic4 = F.interpolate(semantic8, size=shallow4.shape[-2:], mode='bilinear', align_corners=False)
        with torch.cuda.amp.autocast(enabled=False):
            _, enhanced4, aligned4 = self.operator(shallow4.float(), semantic8.float())
            correction = enhanced4 - shallow4.float() + (aligned4 - semantic4.float())
            delta = self.gain * correction
        return shallow4 + semantic4 + delta.to(shallow4.dtype)
