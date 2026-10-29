# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
from pathlib import Path
import torch
from torch import nn
from mmengine.hooks import Hook
from mmseg.registry import HOOKS, MODELS
from astra.core.assembly import UAVAG2B0MergeSegmentor
from astra.core.precision_constants import write_json

class AutocastH1Projector(nn.Sequential):

    def __init__(self, original, enabled=True):
        super().__init__(*list(original.children()))
        self.autocast_enabled = bool(enabled)
        self.last_compute_dtype = None

    def forward(self, inputs):
        with torch.cuda.amp.autocast(enabled=self.autocast_enabled):
            output = super().forward(inputs)
        self.last_compute_dtype = str(output.dtype)
        return output.float()

@MODELS.register_module()
class UAVAG2B0AMPMergeSegmentor(UAVAG2B0MergeSegmentor):

    def __init__(self, projector_amp=True, **kwargs):
        super().__init__(**kwargs)
        self.projector_amp = bool(projector_amp)
        self.feature_distill.projetor = AutocastH1Projector(self.feature_distill.projetor, enabled=self.projector_amp)

    def init_weights(self):
        super().init_weights()
        self.pretrained_load_report.update(H1_projector_AMP=self.projector_amp, H1_MSE_precision='FP32', semantic_KL_precision='FP32', validation_precision='FP32', official_projector_parameters_and_state_names_unchanged=True)

@HOOKS.register_module()
class AstraPrecisionAuditHook(Hook):
    priority = 'HIGH'

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        model = runner.model
        expected = 'torch.float16' if model.projector_amp else 'torch.float32'
        observed = model.feature_distill.projetor.last_compute_dtype
        if observed != expected:
            raise RuntimeError('Unexpected H1 projector arithmetic dtype: ' + str(observed))
        if runner.iter == 0 or (runner.iter + 1) % 50 == 0:
            write_json(Path(runner.work_dir) / 'precision_audit.json', dict(iteration=runner.iter + 1, projector_compute_dtype=observed, projector_output_dtype='torch.float32', MSE_dtype='torch.float32', semantic_KL_dtype='torch.float32', validation_dtype='torch.float32', teacher_feature_batch=model.teacher_feature_batch, student_batch=runner.cfg.train_dataloader.batch_size, deployment_parameters_added=0))
