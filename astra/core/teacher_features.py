# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
from mmseg.registry import MODELS
from astra.core.local_refinement import UAVD4BTransferSegmentor
from astra.core.feature_loss import UAVOmniFeatureLoss

@MODELS.register_module()
class UAVCGLULocalDistillSegmentor(UAVD4BTransferSegmentor):

    def __init__(self, feature_mse_weight, feature_contrast_weight, contrast_enabled, feature_teacher_config, feature_teacher_checkpoint, **kwargs):
        if kwargs.get('transfer_kind') != 'cglu32':
            raise ValueError('This experiment only extends CGLU32')
        super().__init__(**kwargs)
        self.feature_mse_weight = float(feature_mse_weight)
        self.feature_contrast_weight = float(feature_contrast_weight)
        self.feature_teacher_config = feature_teacher_config
        self.feature_teacher_checkpoint = feature_teacher_checkpoint
        with torch.random.fork_rng(devices=[]):
            self.feature_distill = UAVOmniFeatureLoss(contrast_enabled=contrast_enabled)
        object.__setattr__(self, '_feature_teacher', None)
        self.feature_teacher_load_report = None

    def feature_teacher(self, inputs):
        teacher = self._feature_teacher
        if teacher is None:
            from astra.core.teacher_loader import load_model
            device_index = inputs.device.index
            devices = [device_index if device_index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=devices):
                _, teacher, metadata = load_model(self.feature_teacher_config, self.feature_teacher_checkpoint, device=inputs.device)
            if metadata.get('iter') != 28500:
                raise ValueError('Expected original public900B0 iter28500 teacher')
            object.__setattr__(self, '_feature_teacher', teacher)
            self.feature_teacher_load_report = dict(checkpoint=self.feature_teacher_checkpoint, iteration=28500, strict_load=True, requires_grad=False, eval=True, registered_in_student=False, precision='FP16autocast/FP32weights')
        teacher.eval()
        return teacher

    def student_state_dict(self, state_dict=None):
        state = self.state_dict() if state_dict is None else state_dict
        return {name: value for name, value in state.items() if not name.startswith('feature_distill.')}
