"""Strict task-local loading shared by cache generation and native diagnostics."""
from copy import deepcopy
import torch
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmseg.registry import MODELS


def load_model(config_path, checkpoint_path, device='cuda'):
    cfg = Config.fromfile(config_path)
    model_cfg = deepcopy(cfg.model)
    if 'pretrained_checkpoint' in model_cfg:
        model_cfg.pretrained_checkpoint = None
    if 'backbone' in model_cfg and 'pretrained' in model_cfg.backbone:
        model_cfg.backbone.pretrained = None
    model = MODELS.build(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    meta = checkpoint.get('meta', {})
    info = {key: meta[key] for key in ('iter', 'epoch', 'dataset_meta') if key in meta}
    del checkpoint
    model = revert_sync_batchnorm(model).to(device).eval()
    model.requires_grad_(False)
    return cfg, model, info
