# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
from copy import deepcopy
import torch
import torch.nn.functional as F
from mmengine.logging import MMLogger
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.losses.crack_dice_loss import CrackDiceLoss
from mmseg.registry import MODELS
from astra.core.student_factory import load_author_core

@MODELS.register_module()
class UAVAuthorEdgeSegmentor(BaseSegmentor):

    def __init__(self, student_kind, pretrained_checkpoint=None, data_preprocessor=None, train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.student_kind = student_kind
        self.pretrained_checkpoint = pretrained_checkpoint
        self.network = load_author_core(student_kind, checkpoint=pretrained_checkpoint, binary=True)
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or dict(mode='whole')
        self.align_corners = False
        self.num_classes = 2
        self.out_channels = 2
        self.ce = MODELS.build(dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.crack_dice = CrackDiceLoss(loss_weight=1.0, eps=1e-06)
        self.pretrained_load_report = None

    def init_weights(self):
        if self._is_init:
            return
        self.pretrained_load_report = deepcopy(self.network.pretrained_load_report)
        self._is_init = True
        MMLogger.get_current_instance().info('Edge author preload: %s', {k: v for k, v in self.pretrained_load_report.items() if k != 'excluded_training_only_keys'})

    def extract_feat(self, inputs):
        return self.network.backbone(inputs)

    def encode_decode(self, inputs, batch_img_metas=None):
        low = self.network(inputs)
        return F.interpolate(low, size=inputs.shape[-2:], mode='bilinear', align_corners=False)

    def loss(self, inputs, data_samples):
        logits = self.encode_decode(inputs)
        gt = torch.stack([sample.gt_sem_seg.data.squeeze(0) for sample in data_samples]).long()
        if logits.shape[-2:] != gt.shape[-2:]:
            logits = F.interpolate(logits, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        with torch.cuda.amp.autocast(enabled=False):
            logits = logits.float()
            return dict(loss_ce=self.ce(logits, gt, ignore_index=255), loss_crack_dice=self.crack_dice(logits, gt, ignore_index=255))

    def predict(self, inputs, data_samples=None):
        return self.postprocess_result(self.encode_decode(inputs), data_samples)

    def _forward(self, inputs, data_samples=None):
        return self.encode_decode(inputs)
