# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
from pathlib import Path
import math
import torch
from torch import nn
from mmengine.hooks import Hook
from mmseg.registry import MODELS, HOOKS
from astra.core.external_eakd import get_w, kd_loss
from astra.core.detail_distillation import UAVDetailPathReconstructionSegmentor, UAVDetailPathGuard, UAVDetailPathValidationHook
from astra.core.constants import TRAIN_PARAMETERS, DEPLOY_PARAMETERS, write_json

class MeanNormalizedEntropyKD(nn.Module):

    def __init__(self, tau=2.0, entropy_temperature=3.0, loss_weight=1.0):
        super().__init__()
        self.tau = float(tau)
        self.entropy_temperature = float(entropy_temperature)
        self.loss_weight = float(loss_weight)
        self.last_statistics = {}

    def forward(self, logits, gt, margin):
        valid = gt != 255
        if not bool(valid.any()):
            return logits.float().sum() * 0.0
        if margin.requires_grad or margin.shape != gt.shape or logits.shape[1] != 2:
            raise ValueError('EA-KD expects aligned detached binary teacher margins')
        with torch.cuda.amp.autocast(enabled=False):
            student = logits.float().permute(0, 2, 3, 1)[valid]
            delta = margin.detach().float()[valid]
            teacher = torch.stack((torch.zeros_like(delta), delta), dim=1)
            entropy_bound = student.new_tensor(math.log(2.0))
            with torch.no_grad():
                weights = get_w(student, teacher, entropy_bound, t=self.entropy_temperature)
                mean = weights.mean().clamp_min(1e-08)
                self.last_statistics = dict(weight_mean=float(weights.mean()), normalized_weight_min=float(weights.min() / mean), normalized_weight_max=float(weights.max() / mean), entropy_upper_bound=math.log(2.0), valid_pixels=int(valid.sum()))
            loss = kd_loss(student, teacher, temperature=self.tau, EA=True, t=self.entropy_temperature, H_ub=entropy_bound)
            return self.loss_weight * loss / (2.0 * mean)

@MODELS.register_module()
class AstraDistillationSegmentor(UAVDetailPathReconstructionSegmentor):

    def __init__(self, night_objective, scnp_weight=0.25, **kwargs):
        if night_objective != 'EAKD':
            raise ValueError(night_objective)
        super().__init__(**kwargs)
        self.night_objective = night_objective
        self.scnp_weight = float(scnp_weight)
        self._capture_night_logits = False
        self._night_logits = None
        self.last_scnp_loss = None
        self.kd_loss = MeanNormalizedEntropyKD(tau=kwargs.get('kd_tau', 2.0), loss_weight=kwargs.get('kd_weight', 1.0))
        self.architecture_id = 'Astra_' + night_objective + '_unchanged_deployment'

    def init_weights(self):
        super().init_weights()
        self.pretrained_load_report.update(night_objective=self.night_objective, scnp_weight=self.scnp_weight if self.night_objective == 'SCNP' else None, deployment_parameters=DEPLOY_PARAMETERS, deployment_parameters_added=0, inference_graph_changed=False, author_operator_unmodified=True)

    def encode_decode(self, inputs, batch_img_metas=None):
        logits = super().encode_decode(inputs, batch_img_metas)
        if self._capture_night_logits:
            self._night_logits = logits
        return logits

    def loss(self, inputs, data_samples):
        return super().loss(inputs, data_samples)

@HOOKS.register_module()
class AstraObjectiveGuard(UAVDetailPathGuard):

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        outputs = outputs or {}
        model = runner.model
        expected = {'loss_ce', 'loss_crack_dice', 'loss_kd', 'loss_feature_mse', 'loss_semantic_reconstruction'}
        if {k for k in outputs if k.startswith('loss_')} != expected:
            raise RuntimeError('Unexpected overnight objective terms')
        retained = {k: v for k, v in outputs.items() if k != 'loss_scnp'}
        super().after_train_iter(runner, batch_idx, data_batch, retained)
        if self.checked == 1 or self.checked % 50 == 0:
            write_json(Path(runner.work_dir) / 'objective_audit.json', dict(iteration=runner.iter + 1, objective=model.night_objective, loss_terms=sorted(expected), scnp_weight=model.scnp_weight if model.night_objective == 'SCNP' else None, last_scnp_loss=model.last_scnp_loss, entropy_statistics=getattr(model.kd_loss, 'last_statistics', None), original_H1_and_semantic_KD_retained=True, deployment_parameters_added=0, teacher_used_at_inference=False))

@HOOKS.register_module()
class AstraValidationHook(UAVDetailPathValidationHook):

    def before_save_checkpoint(self, runner, checkpoint):
        super().before_save_checkpoint(runner, checkpoint)
        objective = 'Original Astra with mean-normalized EA outputKD; H1 unchanged'
        checkpoint['meta'].update(night_objective=runner.model.night_objective, deployment_parameters_added=0, inference_graph_changed=False, objective=objective, loss=objective)

@HOOKS.register_module()
class AstraFullFitHook(Hook):
    priority = 'LOW'

    def before_train(self, runner):
        model = runner.model
        cfg = runner.cfg
        source = cfg.night_fullfit
        if cfg.val_cfg is not None or cfg.val_dataloader is not None or cfg.val_evaluator is not None:
            raise ValueError('Full fit has no independent validation')
        if len(runner.train_dataloader.dataset) != 1200 or cfg.load_from is not None or cfg.resume:
            raise ValueError('Full fit must start fresh on public1200')
        if sum((p.numel() for p in model.parameters())) != TRAIN_PARAMETERS:
            raise ValueError('Full fit graph differs from selected objective')
        if model.night_objective != source['arm']:
            raise ValueError('Objective mismatch')
        write_json(Path(runner.work_dir) / 'pretrained_load_report.json', model.pretrained_load_report)
        write_json(Path(runner.work_dir) / 'fullfit_audit.json', dict(source, actual_training_images=1200, validation_images=0, fixed_endpoint=runner.max_iters, training_batch=cfg.train_dataloader.batch_size, teacher_feature_batch=model.teacher_feature_batch, teacher_used_at_inference=False, hidden_labels_used=False))

    def before_save_checkpoint(self, runner, checkpoint):
        model = runner.model
        previous = model.training
        model.eval()
        try:
            checkpoint['state_dict'] = model.student_state_dict(checkpoint['state_dict'])
        finally:
            model.train(previous)
        checkpoint.setdefault('meta', {}).update(dict(runner.cfg.night_fullfit), night_objective=model.night_objective, full1200=True, validation_images=0, fixed_endpoint=runner.max_iters, training_batch=runner.cfg.train_dataloader.batch_size, deployment_parameters=DEPLOY_PARAMETERS, teacher_used_at_inference=False, H1_adapter_removed=True, RepDW_folded=True, hidden_labels_used=False)
