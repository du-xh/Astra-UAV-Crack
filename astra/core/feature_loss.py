# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
from astra.core.feature_author import OmniContrastiveFeatureLoss

class UAVOmniFeatureLoss(OmniContrastiveFeatureLoss):

    def __init__(self, contrast_enabled=True):
        super().__init__(student_channels=96, teacher_channels=512, tau_ocd=0.07, M_ocd=16, pool_size=3, patch_size=(4, 4), rand_mask=True, mask_ratio=0.75, enhance_projector=False, dataset='citys')
        self.contrast_enabled = contrast_enabled

    def forward(self, feat_S, feat_T):
        if feat_S.shape[1] != 96 or feat_T.shape[1] != 512:
            raise ValueError('Frozen96-to512 feature contract changed')
        if feat_S.shape[0] != feat_T.shape[0] or feat_S.shape[2:] != feat_T.shape[2:]:
            raise ValueError('Student/teacher feature grids must align')
        if feat_T.shape[-2:] != (48, 84):
            raise ValueError('Pool3/patch4 requires frozen native48x84 grid')
        with torch.cuda.amp.autocast(enabled=False):
            return super().forward(feat_S.float(), feat_T.detach().float())

    def get_omni_contrastive_loss(self, preds, targets):
        if not self.contrast_enabled:
            return preds.sum() * 0.0
        return super().get_omni_contrastive_loss(preds, targets)
