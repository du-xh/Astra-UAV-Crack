# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
from pathlib import Path
import torch
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmseg.registry import HOOKS, MODELS
from astra.core.agreement import ControlledAlignmentFusion, PixelAgreementControl
from astra.core.direction import DirectionContextResidual
from astra.core.calibrated_validation import UAVRepairValidationHook
from astra.core.reparameterization import UAVH1RepDWSegmentor
from astra.core.local_refinement import UAVD4BTransferSegmentor
from astra.core.io import DEPLOY_PARAMETERS, TRAIN_PARAMETERS, write_json

class AG2CGLUForward:

    def add_ag2(self):
        with torch.random.fork_rng(devices=[]):
            self.context = DirectionContextResidual(semantic_guidance=False)
        with torch.random.fork_rng(devices=[]):
            self.fusion = ControlledAlignmentFusion()
        with torch.random.fork_rng(devices=[]):
            self.semantic_control = PixelAgreementControl()
        self.architecture_id = 'AG2_CGLU_RepDW_B0_H1_semanticKD'
        self.inference_forward_batch = 4
        self.inference_image_batch = 8
        self.native_image_batch = 8
        self.tta_view_batch = 4
        self.supervision_only = False

    def encode_decode(self, inputs, batch_img_metas=None):
        features = self.network.backbone(inputs)
        low, middle = self.head_with_middle(features)
        base = F.interpolate(low, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
        semantic8 = self.detail.low_project(middle)
        shallow4 = self.detail.shallow_project(features['stage1'])
        multiplier = self.semantic_control(shallow4, semantic8)
        detail = self.fusion(shallow4, semantic8)
        detail = self.detail.pointwise(self.detail.depthwise(detail))
        detail = self.transfer(detail)
        detail = self.context(detail, None)
        residual = self.detail.residual_logits(detail)
        with torch.cuda.amp.autocast(enabled=False):
            residual = (residual.float() * multiplier).to(residual.dtype)
        return base + F.interpolate(residual, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
    from astra.tta import val_step

@MODELS.register_module()
class UAVAG2B0MergeSegmentor(AG2CGLUForward, UAVH1RepDWSegmentor):

    def __init__(self, teacher_feature_batch=4, **kwargs):
        super().__init__(**kwargs)
        self.add_ag2()
        self.kd_weight = float(kwargs.get('kd_weight', 1.0))
        self.teacher_feature_batch = int(teacher_feature_batch)
        if self.teacher_feature_batch < 1:
            raise ValueError('Teacher feature microbatch must be positive')

    def init_weights(self):
        super().init_weights()
        self.pretrained_load_report.update(architecture=self.architecture_id, teacher='public_train900 B0 iter28500 only', H1=True, semantic_KD=True, RepDW=True, CGLU=True, EMA=False, deployment_parameters=DEPLOY_PARAMETERS, student_training_parameters=TRAIN_PARAMETERS, insertion='FQ3 -> RepDW -> pointwise -> CGLU -> DC1 -> logits -> AG2 multiplier', teacher_feature_batch=self.teacher_feature_batch)

@MODELS.register_module()
class AstraDeploymentSegmentor(AG2CGLUForward, UAVD4BTransferSegmentor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_ag2()
        del self.kd_loss

@HOOKS.register_module()
class UAVAG2MergeGuard(Hook):
    priority = 'VERY_HIGH'

    def before_train(self, runner):
        model = runner.model
        if sum((p.numel() for p in model.parameters())) != TRAIN_PARAMETERS:
            raise RuntimeError('Merged training parameter graph differs from the contract')
        if runner.cfg.load_from is not None or runner.cfg.resume:
            raise RuntimeError('This arm starts fresh from the shared ADE20K pretraining')
        if model.kd_weight != 1.0 or model.semantic_weight != 0.1:
            raise RuntimeError('Keep the retained B0 distillation coefficients')
        self.checked = 0
        self.training_peak_bytes = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        outputs = outputs or {}
        expected = {'loss_ce', 'loss_crack_dice', 'loss_kd', 'loss_feature_mse', 'loss_semantic_reconstruction'}
        actual = {key for key in outputs if key.startswith('loss_')}
        if actual != expected or not bool(torch.isfinite(torch.as_tensor(outputs['loss'])).all()):
            raise FloatingPointError('Invalid merged KD loss: ' + str(actual))
        self.checked += 1
        self.training_peak_bytes = max(self.training_peak_bytes, torch.cuda.max_memory_allocated())
        if self.checked == 1 or self.checked % 50 == 0:
            teacher = runner.model._feature_teacher
            if teacher.training or any((p.requires_grad for p in teacher.parameters())):
                raise RuntimeError('B0 must remain frozen and in evaluation mode')
            write_json(Path(runner.work_dir) / 'distillation_audit.json', dict(checked_iterations=self.checked, observed_loss_terms=sorted(actual), teacher='B0 iter28500 train900', teacher_eval=True, teacher_requires_grad=False, teacher_registered_in_student=False, student_batch=runner.cfg.train_dataloader.batch_size, teacher_feature_batch=runner.model.teacher_feature_batch, training_peak_allocated_bytes=self.training_peak_bytes, H1=True, semantic_KD=True, CGLU=True, RepDW=True, AG2=True, hidden_inputs=False, hidden_labels=False))
            write_json(Path(runner.work_dir) / 'training_memory.json', dict(iteration=runner.iter + 1, training_batch=runner.cfg.train_dataloader.batch_size, peak_cuda_allocated_bytes=self.training_peak_bytes, scope='maximum over training iterations before logger counter resets'))

@HOOKS.register_module()
class UAVAG2MergeValidationHook(UAVRepairValidationHook):
    priority = 'LOWEST'

    def before_train(self, runner):
        self.collapse_count = 0
        write_json(Path(runner.work_dir) / 'pretrained_load_report.json', dict(runner.model.pretrained_load_report, actual_training_batch=runner.cfg.train_dataloader.batch_size, inference_forward_view_batch=4, TTA_forward_calls_per_image=3))

    def before_val_epoch(self, runner):
        super().before_val_epoch(runner)
        runner.model.eval()
        runner.model.enter_deployment()

    def before_save_checkpoint(self, runner, checkpoint):
        model = runner.model
        if model._repdw_unfolded is None:
            raise RuntimeError('Validation and selection must use folded RepDW')
        checkpoint['state_dict'] = model.student_state_dict(checkpoint['state_dict'])
        selected = self._metric(runner).epoch_report['probability_diagnostics']['TTA12']
        checkpoint.setdefault('meta', {}).update(architecture=model.architecture_id, training_batch=runner.cfg.train_dataloader.batch_size, student_training_parameters=TRAIN_PARAMETERS, deployment_parameters=DEPLOY_PARAMETERS, selected_threshold=selected['best_threshold'], selected_score=selected['best_mIoU'], selection_metric='mIoU_TTA12_cal', selection_protocol='public val300 FP32 TTA12 grid0.01:0.01:0.99', teacher_checkpoint=model.feature_teacher_checkpoint, teacher_used_at_inference=False, H1_adapter_removed=True, RepDW_folded=True, TTA_view_batch=4, loss='CE+Dice+B0 outputKD+H1 MSE+semantic reconstruction KL')
        state = checkpoint['state_dict']
        if any((key.startswith('feature_distill.') for key in state)):
            raise RuntimeError('H1 adapter leaked into deployment checkpoint')
        if any((key.startswith('detail.depthwise.conv.conv') for key in state)):
            raise RuntimeError('RepDW training branches leaked into deployment checkpoint')

    def after_val_epoch(self, runner, metrics=None):
        try:
            super().after_val_epoch(runner, metrics)
        finally:
            runner.model.exit_deployment()
