# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
import torch.nn.functional as F
from astra.core.feature_author import OmniContrastiveFeatureLoss

class UAVHighResolutionReconstruction(OmniContrastiveFeatureLoss):

    def __init__(self, target_mode):
        if target_mode not in ['pooled_target', 'raw_target']:
            raise ValueError('Unknown paired teacher target: ' + str(target_mode))
        super().__init__(student_channels=96, teacher_channels=512, tau_ocd=0.07, M_ocd=16, pool_size=3, patch_size=(4, 4), rand_mask=True, mask_ratio=0.75, enhance_projector=False, dataset='citys')
        self.target_mode = target_mode
        self.contrast_enabled = False

    def make_target(self, feature):
        if feature.shape[1:] != (512, 96, 168):
            raise ValueError('Expected native padded teacher512/stride4 grid96x168')
        feature = feature.detach().float()
        if self.target_mode == 'pooled_target':
            return F.interpolate(F.avg_pool2d(feature, 2, 2), scale_factor=2, mode='nearest')
        return feature

    def forward(self, student, teacher):
        if student.shape[1:] != (96, 48, 84) or student.shape[0] != teacher.shape[0]:
            raise ValueError('Expected student96/stride8 grid48x84 and aligned batch')
        with torch.cuda.amp.autocast(enabled=False):
            target = self.make_target(teacher)
            student = F.interpolate(student.float(), size=target.shape[-2:], mode='bilinear', align_corners=False)
            return super().forward(student, target)

    def get_omni_contrastive_loss(self, predictions, targets):
        return predictions.sum() * 0.0
