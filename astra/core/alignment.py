# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
from torch import nn
import torch.nn.functional as F
from astra.core.freqfusion import FreqFusion
from astra.core.fusion_base import ConservativeFrequencyFusion
OPTIONS = dict(hr_channels=32, lr_channels=32, compressed_channels=16, scale_factor=1, lowpass_kernel=5, highpass_kernel=3, up_group=1, encoder_kernel=3, upsample_mode='nearest', align_corners=False, feature_resample=True, feature_resample_group=4, comp_feat_upsample=True, use_high_pass=True, use_low_pass=True, hr_residual=True, semi_conv=True, hamming_window=False, feature_resample_norm=True)

class LocalSimilarityAlignmentOnly(FreqFusion):
    """Retain the author's compressors/sampler and their initial parameters."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.content_encoder
        del self.content_encoder2
        del self.hamming_lowpass
        del self.hamming_highpass
        self.use_high_pass = False
        self.use_low_pass = False
        self.comp_feat_upsample = False

    def _forward(self, hr_feat, lr_feat):
        compressed_hr = self.hr_channel_compressor(hr_feat)
        compressed_lr = self.lr_channel_compressor(lr_feat)
        values = F.interpolate(lr_feat, size=hr_feat.shape[-2:], mode='bilinear', align_corners=False)
        aligned = self.dysampler(hr_x=compressed_hr, lr_x=compressed_lr, feat2sample=values)
        return (None, hr_feat, aligned)

class SplitFrequencyFusion(ConservativeFrequencyFusion):
    """Use the unchanged FQ1 residual adapter with a reduced author operator."""

    def __init__(self, kind):
        nn.Module.__init__(self)
        self.operator = LocalSimilarityAlignmentOnly(**OPTIONS)
        self.gain = nn.Parameter(torch.full((1, 32, 1, 1), 0.01))
