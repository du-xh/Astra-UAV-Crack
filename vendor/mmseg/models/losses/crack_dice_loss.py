import torch
import torch.nn as nn

from mmseg.registry import MODELS


@MODELS.register_module()
class CrackDiceLoss(nn.Module):
    """Dice loss calculated only for the crack class (class index 1)."""

    def __init__(
        self,
        loss_weight=1.0,
        eps=1e-6,
        loss_name="loss_crack_dice",
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name

    def forward(
        self,
        pred,
        target,
        weight=None,
        ignore_index=255,
        **kwargs,
    ):
        # pred: N x 2 x H x W
        # target: N x H x W
        if target.ndim == 4:
            target = target.squeeze(1)

        crack_prob = torch.softmax(pred, dim=1)[:, 1]
        crack_target = (target == 1).to(dtype=crack_prob.dtype)
        valid_mask = target != ignore_index

        if weight is not None:
            valid_weight = weight.to(dtype=crack_prob.dtype)
        else:
            valid_weight = torch.ones_like(crack_prob)

        valid_weight = valid_weight * valid_mask.to(
            dtype=crack_prob.dtype
        )

        crack_prob = crack_prob * valid_weight
        crack_target = crack_target * valid_weight

        # 在整个batch上计算前景Dice，避免无裂缝patch导致不稳定
        intersection = torch.sum(crack_prob * crack_target)
        denominator = torch.sum(crack_prob) + torch.sum(crack_target)

        dice = (
            2.0 * intersection + self.eps
        ) / (
            denominator + self.eps
        )

        return self.loss_weight * (1.0 - dice)

    @property
    def loss_name(self):
        return self._loss_name