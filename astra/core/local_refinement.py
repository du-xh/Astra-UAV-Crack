# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
from pathlib import Path
import torch
from torch import nn
from mmseg.registry import MODELS
from mmengine.logging import MMLogger
from astra.core.detail_branch import UAVInnovation20261016Segmentor
from astra.core.cglu import ConvolutionalGLU

class DetailCGLU(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(32, eps=1e-06)
        self.cglu = ConvolutionalGLU(32, hidden_features=96, out_features=32, drop=0.0)
        self.scale = nn.Parameter(torch.zeros(32))

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        delta = self.cglu(self.norm(tokens), h, w) * self.scale
        return x + delta.transpose(1, 2).reshape(b, c, h, w)

@MODELS.register_module()
class UAVD4BTransferSegmentor(UAVInnovation20261016Segmentor):

    def __init__(self, transfer_kind='control', transfer_init_checkpoint=None, **kwargs):
        if transfer_kind != 'cglu32':
            raise ValueError('Unknown D4-B transfer: ' + transfer_kind)
        if kwargs.get('variant', 'D4-B') != 'D4-B':
            raise ValueError('Only the final stride4 detail branch is packaged.')
        if kwargs.get('initial_checkpoint') is not None:
            raise ValueError('Use transfer_init_checkpoint for a full trained D4-B state.')
        if transfer_init_checkpoint is not None and kwargs.get('pretrained_checkpoint') is not None:
            raise ValueError('Do not reload ADE when strictly continuing the trained D4-B model.')
        kwargs['variant'] = 'D4-B'
        super().__init__(**kwargs)
        self.transfer_kind = transfer_kind
        self.transfer_load_report = None
        if transfer_init_checkpoint is not None:
            checkpoint = torch.load(str(Path(transfer_init_checkpoint)), map_location='cpu', weights_only=False)
            self.load_state_dict(checkpoint['state_dict'], strict=True)
            self.transfer_load_report = dict(path=str(transfer_init_checkpoint), strict=True, source_iteration=checkpoint.get('meta', {}).get('iter'), original_tensors=len(checkpoint['state_dict']))
            self.continuation_load_report = self.transfer_load_report
            del checkpoint
        with torch.random.fork_rng(devices=[]):
            self.transfer = DetailCGLU()

    def init_weights(self):
        if self._is_init:
            return
        super().init_weights()
        MMLogger.get_current_instance().info('D4-B transfer=%s; complete trained-state load=%s; no ADE reload', self.transfer_kind, self.transfer_load_report)
