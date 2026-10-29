# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
from mmseg.registry import HOOKS, MODELS
from astra.core.reconstruction_author import OmniContrastiveFeatureLoss
from astra.core.precision import UAVAG2B0AMPMergeSegmentor, AutocastH1Projector
from astra.core.assembly import UAVAG2MergeGuard, UAVAG2MergeValidationHook
from astra.core.highres_loss import UAVHighResolutionReconstruction
from astra.core.semantic_distillation import reconstruction_semantic_kl
from astra.core.local_refinement import UAVD4BTransferSegmentor
from astra.core.precision_constants import TRAIN_PARAMETERS, DEPLOY_PARAMETERS

class DetailPathReconstruction(OmniContrastiveFeatureLoss):

    def __init__(self, target_mode):
        super().__init__(student_channels=32, teacher_channels=512, tau_ocd=0.07, M_ocd=16, pool_size=3, patch_size=(4, 4), rand_mask=True, mask_ratio=0.75, enhance_projector=False, dataset='citys')
        self.target_mode = target_mode
        self.contrast_enabled = False
    make_target = UAVHighResolutionReconstruction.make_target

    def forward(self, student, teacher):
        if student.shape[1:] != (32, 96, 168) or student.shape[0] != teacher.shape[0]:
            raise ValueError('Expected gated detail32/stride4 at the native padded grid')
        with torch.cuda.amp.autocast(enabled=False):
            return super().forward(student.float(), self.make_target(teacher))

    def get_omni_contrastive_loss(self, predictions, targets):
        return predictions.sum() * 0.0

@MODELS.register_module()
class UAVDetailPathReconstructionSegmentor(UAVAG2B0AMPMergeSegmentor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        retained_projector = self.feature_distill.projetor.state_dict()
        with torch.random.fork_rng(devices=[]):
            self.feature_distill = DetailPathReconstruction(kwargs['feature_target_mode'])
        self.feature_distill.projetor.load_state_dict(retained_projector, strict=True)
        self.feature_distill.projetor = AutocastH1Projector(self.feature_distill.projetor, enabled=self.projector_amp)
        self.architecture_id = 'AG2_CGLU_RepDW_DPR1_gated_detail4_B0KD'

    def init_weights(self):
        super().init_weights()
        self.pretrained_load_report.update(architecture=self.architecture_id, student_training_parameters=TRAIN_PARAMETERS, deployment_parameters=DEPLOY_PARAMETERS, reconstruction_source='AG2 multiplier * post-CGLU/DC1 detail32/stride4', reconstruction_source_channels=32, reconstruction_source_stride=4, reconstruction_target='unchanged pooled_target B0 teacher512/stride4', teacher_source_changed=False, inference_graph_changed=False)

    def loss(self, inputs, data_samples):
        captured = {}
        detail_hook = self.detail.residual_logits.register_forward_pre_hook(lambda module, args: captured.update(detail=args[0]))
        gate_hook = self.semantic_control.register_forward_hook(lambda module, args, output: captured.update(multiplier=output))
        try:
            losses = UAVD4BTransferSegmentor.loss(self, inputs, data_samples)
        finally:
            detail_hook.remove()
            gate_hook.remove()
        with torch.cuda.amp.autocast(enabled=False):
            student = captured['detail'].float() * captured['multiplier'].float()
        teacher = self.feature_teacher(inputs)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            feature = torch.cat([teacher.decode_head._forward_feature(teacher.extract_feat(chunk)) for chunk in inputs.split(self.teacher_feature_batch)], dim=0)
        projector_hook = self.feature_distill.projetor.register_forward_hook(lambda module, args, output: captured.update(reconstruction=output))
        try:
            mse, _ = self.feature_distill(student, feature)
        finally:
            projector_hook.remove()
        losses['loss_feature_mse'] = self.feature_mse_weight * mse
        with torch.no_grad():
            target = self.feature_distill.make_target(feature)
        valid = torch.stack([sample.gt_sem_seg.data.squeeze(0) for sample in data_samples]) != 255
        losses['loss_semantic_reconstruction'] = self.semantic_weight * reconstruction_semantic_kl(captured['reconstruction'], target, teacher.decode_head, valid, self.semantic_temperature)
        self.last_source_shape = list(student.shape)
        return losses

@HOOKS.register_module()
class UAVDetailPathGuard(UAVAG2MergeGuard):

    def before_train(self, runner):
        model = runner.model
        if sum((p.numel() for p in model.parameters())) != TRAIN_PARAMETERS:
            raise RuntimeError('Unexpected detail-path training graph')
        if runner.cfg.load_from is not None or runner.cfg.resume:
            raise RuntimeError('DPR1 starts fresh from the shared ADE20K pretraining')
        if model.feature_distill.target_mode != 'pooled_target' or model.kd_weight != 1.0 or model.semantic_weight != 0.1:
            raise RuntimeError('Teacher target and distillation coefficients must remain fixed')
        self.checked = 0
        self.training_peak_bytes = 0

@HOOKS.register_module()
class UAVDetailPathValidationHook(UAVAG2MergeValidationHook):

    def before_save_checkpoint(self, runner, checkpoint):
        super().before_save_checkpoint(runner, checkpoint)
        checkpoint['meta'].update(student_training_parameters=TRAIN_PARAMETERS, distillation_feature_source='gated detail32/stride4', distillation_target='unchanged pooled B0 feature512/stride4', training_parameter_delta_vs_merge=-32768, deployment_parameters_added=0, inference_graph_changed=False)
