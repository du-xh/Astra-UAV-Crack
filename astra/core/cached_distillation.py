# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from astra.core.base_student import UAVAuthorEdgeSegmentor

@MODELS.register_module()
class UAVCachedKD20261015Segmentor(UAVAuthorEdgeSegmentor):

    def __init__(self, kd_mode='ordinary', kd_tau=2.0, kd_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.kd_mode = kd_mode
        self.kd_loss = torch.nn.Identity()  # Replaced by entropy KD in the final training class.

    def loss(self, inputs, data_samples):
        logits = self.encode_decode(inputs)
        gt = torch.stack([s.gt_sem_seg.data.squeeze(0) for s in data_samples]).long()
        if logits.shape[-2:] != gt.shape[-2:]:
            raise ValueError('KD baseline must retain the frozen native output geometry')
        margins = []
        for sample in data_samples:
            margin = sample.teacher_margin.data.squeeze(0)
            h, w = margin.shape
            if h > gt.shape[-2] or w > gt.shape[-1]:
                raise ValueError('Teacher exceeds the padded GT geometry')
            margins.append(F.pad(margin, (0, gt.shape[-1] - w, 0, gt.shape[-2] - h)))
        margin = torch.stack(margins).detach()
        with torch.cuda.amp.autocast(enabled=False):
            logits = logits.float()
            return dict(loss_ce=self.ce(logits, gt, ignore_index=255), loss_crack_dice=self.crack_dice(logits, gt, ignore_index=255), loss_kd=self.kd_loss(logits, gt, margin))
